#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include "ggml-onednn.h"

using namespace dnnl;

static inline void copy_to_onednn(const ggml_tensor * tensor, memory &mem);
static inline void copy_from_onednn(const ggml_tensor * tensor, memory &mem);

static void better_assert(bool cond, const char * str = nullptr){
    if (cond == false){
        str = str ? str : "better_assert(): assert failed";
        fprintf(stderr, "%s\n", str);
        abort();
    }
}

static engine * get_onednn_engine(){
    static engine * eng = nullptr;
    if (eng == nullptr) eng = new engine(engine::kind::cpu, 0);
    return eng;
}

static stream * get_onednn_stream(){
    static stream * str = nullptr;
    if (str == nullptr) str = new stream(*get_onednn_engine());
    return str;
}

static dnnl::memory::data_type convert_type_ggml_to_onednn(const enum ggml_type type){
    using dt = dnnl::memory::data_type;
    
    switch (type){
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_Q8_0:
            return dt::s8;
        case GGML_TYPE_F16:
            return dt::f16;
        
        default:
            printf("Other type detected! typeid: %d\n", type);
            abort();
            return dt::undef;
    }
}

static bool get_type_quantized(dnnl::memory::data_type type){
    using dt = dnnl::memory::data_type;
    switch (type){
        case dt::f32:
        case dt::f16:
            return false;
        case dt::s8:
            return true;
        
        default:
            better_assert(false, "unexpected type in get_type_quantized");
            return true;
    }
}

static bool get_type_blocked(const enum ggml_type type){
    switch (type){
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return false;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return true;
        
        default:
            better_assert(false, "unexpected type in get_type_blocked");
            return true;
    }
}

static bool valid_weight_type(ggml_type type){
    switch (type){
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        {
            return true;
        }
        default:
        {
            return false;
        }
    }
}

static memory::data_type get_matmul_input_dt(ggml_type type){
    using dt = memory::data_type;
    switch (type){
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_F16:
            return dt::f16;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return dt::s8;
        default:
            better_assert(false, "Invalid type in get_matmul_input_dt");
            return dt::undef;
    }
}

static memory::data_type get_matmul_weights_dt(ggml_type type){
    using dt = memory::data_type;
    switch (type){
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_F16:
            return dt::f16;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return dt::s8;
        default:
            better_assert(false, "Invalid type in get_matmul_input_dt");
            return dt::undef;
    }
}

static memory::data_type get_matmul_output_dt(ggml_type type){
    using dt = memory::data_type;
    switch (type){
        case GGML_TYPE_F32:
            return dt::f32;
        case GGML_TYPE_F16:
            return dt::f16;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
            return dt::f32;
        default:
            better_assert(false, "Invalid type in get_matmul_output_dt");
            return dt::undef;
    }
}

float clamp(float val, float min, float max){
    return std::max(std::min(val, max), min);
}

struct memory_maybe_quant{
    bool quantized = false;
    memory mem;
    memory scale;
    
    memory_maybe_quant(){}
    memory_maybe_quant(memory _mem){mem = _mem;}
    memory_maybe_quant(memory _mem, memory _scale){mem = _mem; scale = _scale; quantized = true;}
};

struct optional_memory{
    memory_maybe_quant handle;
    bool valid;
    optional_memory(memory_maybe_quant mem){handle = mem; valid = true;}
    optional_memory(){valid = false;}
};

memory_maybe_quant requantize_weights(const struct ggml_tensor * weights){
    using dt = memory::data_type;
    using tag = memory::format_tag;
    auto &engine = *get_onednn_engine();
    
    memory::dims weights_dims = {weights->ne[3], weights->ne[2], weights->ne[0], weights->ne[1]}; // Note that the 3,2,0,1 ordering is intentional because the weights are transpsosed in storage
    auto weights_type = get_matmul_weights_dt(weights->type);
    auto weights_tag = tag::abdc; // Note that this is intentionally a transposed tag
    auto weights_md = memory::desc(weights_dims, weights_type, weights_tag); 
    auto scale_md = memory::desc({1}, dt::f32, tag::x);
    
    auto weights_mem = memory(weights_md, engine);
    auto scale_mem = memory(scale_md, engine);
    
    // We calculate the scale, requantize as 
    float scale = 0.0f;
    
    if (get_type_blocked(weights->type)){
        auto traits = ggml_internal_get_type_traits(weights->type);
        int64_t blockSize = traits.blck_size;
        
        void * blocks = weights->data;
        int8_t * ptr = (int8_t *) weights_mem.get_data_handle();
        
        size_t elemCount = weights->ne[0] * weights->ne[1] * weights->ne[2] * weights->ne[3];
        size_t blockCount = elemCount / blockSize;

        better_assert(traits.to_float != NULL, "weights type does not support to_float");
        better_assert(elemCount % blockSize == 0, "elements not multiple of block size");
        
        std::vector<float> blockData(blockSize);
        
        // Calculate average scale
        
        //float absmax = 0.0f;
        
        for (int blockID = 0; blockID < blockCount; blockID++){
            auto * block = blocks + blockID * traits.type_size;
            traits.to_float(block, blockData.data(), 1 * blockSize); // last 1 is # of values to dequant, currently 1 blocks worth
            
            float absmax = 0.0f;
            for (int subIndex = 0; subIndex < blockSize; subIndex++){
                absmax = std::max(absmax, std::abs(blockData[subIndex]));
            }
            
            scale += absmax / 127;
            //printf("scale val: %f, calced new scale: %f\n", ggml_fp16_to_fp32(blocks[blockID].d), absmax / 127);
        }
        
        //printf("Final absmax: %f\n", absmax);
        
        //scale = absmax / 127;
        scale = scale / blockCount;
        
        if (scale == 0){
        for (int blockID = 0; blockID < blockCount; blockID++){
            auto * block = blocks + blockID * traits.type_size;
            traits.to_float(block, blockData.data(), 1 * blockSize); // last 1 is # of values to dequant, currently 1 blocks worth
            for (int subIndex = 0; subIndex < blockSize; subIndex++){
                if (blockData[subIndex] != 0){
                    printf("What on earth?\n");
                    abort();
                }
            }
        }
            
            // All values are a constant 0? Okay I guess
            printf("All weights zero case hit\n");
            scale = 0.0f;
            for (int i = 0; i < elemCount; i++){
                ptr[elemCount] = 0;
            }
        } else {
        
            float invscale = 1.0 / scale;
            //printf("scale: %f, invscale: %f\n", scale, invscale);
            
            better_assert(invscale != 0, "invscale == 0, should not be possible");
            better_assert(scale != 0, "scale == 0, not currently handled");
            better_assert(scale > 0, "scale is negative, should not be possible");
            
            // Now work through the blocks, dequantizing and requantizing as needed
            
            int i = 0;
            
            int weirds = 0;
            
            for (int blockID = 0; blockID < blockCount; blockID++){
                auto * block = blocks + blockID * traits.type_size;
                traits.to_float(block, blockData.data(), 1 * blockSize); // last 1 is # of values to dequant, currently 1 blocks worth
                for (int subIndex = 0; subIndex < blockSize; subIndex++){
                    ptr[i] = clamp(roundf(blockData[subIndex] * invscale), -128, 127);
                    if (ptr[i] == -128 || ptr[i] == 127 || (ptr[i] == 0 && blockData[subIndex] != 0)) weirds++;
                    i++;
                }
            }
            
            //printf("Weights Weirds % = %f\n", 100.0 * weirds / elemCount);
        }
    }
    

    *((float *)scale_mem.get_data_handle()) = scale;
    
    return memory_maybe_quant(weights_mem, scale_mem);
}

optional_memory try_prepare_weights(const struct ggml_tensor * weights){
    using dt = memory::data_type;
    using tag = memory::format_tag;
    auto &engine = *get_onednn_engine();
    
    memory::dims weights_dims = {weights->ne[3], weights->ne[2], weights->ne[0], weights->ne[1]}; // Note that the 3,2,0,1 ordering is intentional because the weights are transposed in storage
    auto weights_type = get_matmul_weights_dt(weights->type);
    auto weights_tag = tag::abdc; // Note that this is intentionally a transposed tag
    auto weights_md = memory::desc(weights_dims, weights_type, weights_tag); 
    auto mem = memory(weights_md, engine);
    
    if (!get_type_blocked(weights->type) && ggml_is_contiguous(weights)){
        // We can use copyless logic
        mem.set_data_handle(weights->data);
        
        return optional_memory(memory_maybe_quant(mem));
    }else if (!get_type_blocked(weights->type) && ggml_is_contiguous(weights)){ // TODO does ggml_is_contiguous already check for being blocked?
        // We can make a copyless buffer, than use a reorder to copy it
        
        // TODO this makes no sense to me, is it me or the code?
        // (why are we reading it as f32? why bother with any of this???)
        // (wouldnt this only be needed if the type of weights we are going to use doesn't match the type of weights in storage, which I don't think ever happens?)
        
        printf("This code path makes no sense to me, I made it always fail. Reorder weights path\n");
        abort();
        /*auto &stream = *get_onednn_stream();
        
        auto tensor_md  = memory::desc(weights_dims, dt::f32, weights_tag);
        auto tensor_mem = memory(tensor_md, engine);
        
        tensor_mem.set_data_handle(weights->data);
        
        reorder(tensor_mem, mem).execute(stream, tensor_mem, mem);
        stream.wait();
        
        return optional_memory(mem);*/
    }else if (get_type_blocked(weights->type)){
        // Dequantize and requantize weight types
        //printf("Dequantize logic, unimplemented\n");
        
        return optional_memory(requantize_weights(weights));
    }else{
        // We have to copy it using specific logic
        
        printf("Not sure this codepath works!!!\n");
        copy_to_onednn(weights, mem);
        
        return optional_memory(memory_maybe_quant(mem));
    }
    
    return optional_memory();
}


memory_maybe_quant setup_input_onednn(const struct ggml_tensor * src, ggml_type weightsType){
    using dt = memory::data_type;
    using tag = memory::format_tag;
    auto &engine = *get_onednn_engine();
    
    auto type = get_matmul_input_dt(weightsType);
    
    memory::dims srcdims  = {src->ne[3],  src->ne[2],  src->ne[1],  src->ne[0]};
    auto src_md  = memory::desc(srcdims, type, tag::abcd);
    auto mem = memory(src_md, engine);
    
    if (!get_type_quantized(type)){
        if (type == dt::f32 && ggml_is_contiguous(src)){
            // We can do copyless
            mem.set_data_handle(src->data);
        }else{
            // We cannot do copyless
            // Can do fast copy?
            if (ggml_is_contiguous(src)){
                auto &stream = *get_onednn_stream();
                
                auto tensor_md  = memory::desc(srcdims, dt::f32, tag::abcd);
                auto tensor_mem = memory(tensor_md, engine);
                
                tensor_mem.set_data_handle(src->data);
                
                reorder(tensor_mem, mem).execute(stream, tensor_mem, mem);
                stream.wait();
                
            }else{
                // uhoh
                printf("TODO unimplemented slow copy for setup_input_onednn\n");
                abort();
            }
        }
    }else{
        // We must first quantize the input
        auto scale_md = memory::desc({1}, dt::f32, tag::x);
        auto scale_mem = memory(scale_md, engine);
        
        if (type == dt::s8){
            size_t inputSize = src->ne[0] * src->ne[1] * src->ne[2] * src->ne[3];
            float absmax = 0;
            
            float * srcData = (float *)src->data;
            
            int count = 0;
            for (size_t i = 0; i < inputSize; i++){
                absmax = std::max(std::abs(srcData[i]), absmax);
            }
            
            int8_t * onednnData = (int8_t *) mem.get_data_handle();
            
            int sumtot = 0;
            
            // Calculate min/max/percentiles
            std::vector<float> values; values.resize(inputSize); for(int i = 0;i < inputSize; i++) values[i] = std::abs(srcData[i]); std::sort(values.begin(), values.end());
            float mini = values[0];
            float maxi = values[inputSize - 1];
            float perc10 = values[roundf(inputSize * 0.1)];
            float perc90 = values[roundf(inputSize * 0.9)];
            float median = values[roundf(inputSize * 0.5)];
            
            float perc1 = values[roundf(inputSize * 0.01)];
            float perc99 = values[roundf(inputSize * 0.99)];
            float perc999 = values[roundf(inputSize * 0.999)];
            
            //printf("perc99: %f, perc999: %f, maxi: %f\n", perc99, perc999, maxi);
            //printf("99-01perc delta: %f, max-min: %f\n", perc99 - perc1, maxi - mini);
            
            // Just for testing HERE HERE HERE ONEDNN
            //absmax = perc999;
            
            //printf("\n00%: %f, 01%: %f, 10%: %f, 50%: %f, 90%: %f, 99%: %f, 100%: %f\n", mini, perc1, perc10, median, perc90, perc99, maxi);
            
            int capCount = 0;
            int zeroCount = 0;
            
            float scale = absmax / 127;
            if (scale != 0){
                better_assert(scale != 0, "scale == 0 in input quantization, not currently supported.");
                float invscale = 1.0 / scale;
                
                for (size_t i = 0; i < inputSize; i++){
                    int8_t value = roundf(clamp(invscale * srcData[i], -128, 127));
                    sumtot += std::abs(value);
                    if (value == -128 || value == 127 || (value == 0 && srcData[i] != 0)) capCount++;
                    if (value == 0 && srcData[i] != 0) zeroCount++;
                    //printf("value: %f, invscale: %f, non-rounded: %f, rounded value: %d\n", srcData[i], invscale, invscale * srcData[i], value);
                    onednnData[i] = value;
                }
            }else{
                for (size_t i = 0; i < inputSize; i++){
                    onednnData[i] = 0;
                }
            }
            
            //printf("Avg. abs: %f\n", (float)sumtot / inputSize);
            
            printf("oneDNN input quants weirds % = %f, zero weirds % = %f\n", 100.0 * (float)capCount / inputSize, 100.0 * (float)zeroCount / inputSize); 
            
            *((float *)scale_mem.get_data_handle()) = scale;

        }else{
            printf("Unexpected quantized type in setup_input_onednn! Aborting...\n");
            abort();
        }
        
        return memory_maybe_quant(mem, scale_mem);
    }
    
    return memory_maybe_quant(mem);
}

memory setup_output_onednn(const struct ggml_tensor * dst, ggml_type weightsType, bool &copyless){
    using dt = memory::data_type;
    using tag = memory::format_tag;
    auto &engine = *get_onednn_engine();
    
    auto type = get_matmul_output_dt(weightsType);
    
    memory::dims dstdims  = {dst->ne[3],  dst->ne[2],  dst->ne[1],  dst->ne[0]};
    auto dst_md  = memory::desc(dstdims, type, tag::abcd);
    auto mem = memory(dst_md, engine);

    if (type == dt::f32 && ggml_is_contiguous(dst)){
        // We can do copyless
        copyless = true;
        
        mem.set_data_handle(dst->data);
    }else{
        // We cannot do copyless
        copyless = false;
    }
    
    return mem;
}

void copy_output_onednn(const struct ggml_tensor * dst, memory mem){
    using tag = memory::format_tag;

    if (ggml_is_contiguous(dst)){
        auto &engine = *get_onednn_engine();
        auto &stream = *get_onednn_stream();
        
        auto tensor_type = convert_type_ggml_to_onednn(dst->type);
        auto tensor_md  = memory::desc(mem.get_desc().get_dims(), tensor_type, tag::abcd);
        auto tensor_mem = memory(tensor_md, engine);
        
        tensor_mem.set_data_handle(dst->data);
        
        reorder(mem, tensor_mem).execute(stream, mem, tensor_mem);
        stream.wait();
    }else{
        printf("TODO unimplemented slow copy in copy_output_onednn");
    }
}

extern "C" bool ggml_try_onednn_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, const int ith, const int nth){
    if (src0->nb[2] != src0->nb[3] || src1->nb[2] != src1->nb[3] || dst->nb[2] != dst->nb[3]) return false;

    
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32){
        printf("Unexpected source/destination type! Not using oneDNN here\n");
        return false;
    }
    if (!valid_weight_type(src0->type)){
        printf("NOTE: Invalid weight type %d\n", src0->type);
        return false;
    }
    
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, memory_maybe_quant> cache;
    
    // #1: Check if the weights are cached
    
    std::string weights_name = src0->name;
    
    cache_mutex.lock();
    auto result = cache.find(weights_name);
    if (result != cache.end()){
        // Cache hit
        // Don't do anything else here
    }else{
        // Cache miss
        auto new_memory = try_prepare_weights(src0);
        if (new_memory.valid){
            cache.emplace(weights_name, new_memory.handle);
        }else{
            // Probably shouldn't get here, since we check for a valid weight type at the beginning
            cache_mutex.unlock();
            printf("Unexpected failure to add new weight to cache, aborting...\n");
            abort();
            return false;
        }
        cache_mutex.unlock();
    }
    
    auto matmul_weights_holder = cache[weights_name];
    auto matmul_weights = matmul_weights_holder.mem;
    
    cache_mutex.unlock();
    
    // If we get here, we should be doing the matrix multiplication; failing after this point should be a serious bug
    if (ith != 0) return true;
    
    // Copy input to onednn
    auto matmul_input_holder = setup_input_onednn(src1, src0->type);
    auto matmul_input = matmul_input_holder.mem;
    
    // Prepare output buffer
    bool copyless = false;
    auto matmul_output = setup_output_onednn(dst, src0->type, copyless);

    // Run MatMul primitive
    auto &engine = *get_onednn_engine();
    auto &stream = *get_onednn_stream();
    
    better_assert(matmul_weights_holder.quantized == matmul_input_holder.quantized, "Inputs and weights quantization do not match, not currently supported");
    
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, matmul_input});
    matmul_args.insert({DNNL_ARG_WEIGHTS, matmul_weights});
    matmul_args.insert({DNNL_ARG_DST, matmul_output});
    
    primitive_attr matmul_attr;
    if (matmul_weights_holder.quantized){
        int data_mask = 0; // I don't know what these masks mean, but they should be 0
        matmul_attr.set_scales_mask(DNNL_ARG_SRC, data_mask);
        matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS, data_mask);
        
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, matmul_input_holder.scale});
        matmul_args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, matmul_weights_holder.scale});
    }
    
    auto matmul_pd = matmul::primitive_desc(engine,  matmul_input.get_desc(), matmul_weights.get_desc(), matmul_output.get_desc(), matmul_attr);
    auto matmul_prim = matmul(matmul_pd);
    
    matmul_prim.execute(stream, matmul_args);
    
    stream.wait();
    
    // Load data back out of onednn, if needed converting from the output type to f32 (note that the desired output is always f32, checked at top of function)
    if (!copyless){
        // Couldn't do a copyless setup, so we have to copy the results back out
        copy_output_onednn(dst, matmul_output);
    }
    
    return true;
}


// Read from memory, write to handle
static inline void copy_from_onednn(const ggml_tensor * tensor, memory &mem) {
    auto handle = tensor->data;
    void *ptr = mem.get_data_handle();
    size_t bytes = mem.get_desc().get_size();
    
    if (ptr) {
        for (size_t i = 0; i < bytes; ++i) {
            ((char *)handle)[i] = ((char *)ptr)[i];
        }
    }
}

// Read from handle, write to memory
static inline void copy_to_onednn(const ggml_tensor * tensor, memory &mem) {
    auto handle = tensor->data;
    size_t bytes = mem.get_desc().get_size();
    
    if (!get_type_blocked(tensor->type)){
        void *ptr = mem.get_data_handle();

        switch (tensor->type){
        case GGML_TYPE_F32:
        {
            if (ptr) {
                better_assert(bytes % 4 == 0, "bytes not divisible by 4");
                for (size_t i = 0; i < bytes / 4; ++i) {
                    ((float *)ptr)[i] = ((float *)handle)[i];
                }
            }
            break;
        }
        case GGML_TYPE_F16:
        {
            if (ptr) {
                better_assert(bytes % 4 == 0, "bytes not divisible by 4");
                for (size_t i = 0; i < bytes / 4; ++i) {
                    ((float *)ptr)[i] = ggml_fp16_to_fp32(((int16_t *)handle)[i]);
                }
            }
            break;
        }
        default:
        {
            better_assert(false, "Invalid non-blocked type in copy_to_onednn");
        }
        }
    }else{
        if (tensor->type == GGML_TYPE_Q8_0){
            int blockSize = 32;
            block_q8_0 * blocks = (block_q8_0 *) handle;
            size_t elemCount = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
            size_t blockCount = elemCount / blockSize;

            better_assert(elemCount % blockSize == 0, "elements not multiple of block size, ah jeez");
            
            float * fPtr = (float *) mem.get_data_handle();
            
            float mean = 0;
            for (size_t blockID = 0; blockID < blockCount; blockID++){
                mean += ggml_fp16_to_fp32(blocks[blockID].d);
            }
            mean /= blockCount;
            
            double stddev = 0;
            for (size_t blockID = 0; blockID < blockCount; blockID++){
                double factor = ggml_fp16_to_fp32(blocks[blockID].d);
                stddev += (factor - mean) * (factor - mean);
            }
            stddev /= blockCount;
            stddev = std::sqrt(stddev);
            printf("mean: %f, STD: %f\n", mean, stddev);
            
            for (size_t blockID = 0; blockID < blockCount; blockID++){
                block_q8_0 block = blocks[blockID];

                float factor = ggml_fp16_to_fp32(block.d);
                if (factor < 0.0){
                    // Negative factor, uhoh!
                    printf("Unexpected negative factor in q8 dequantization\n");
                    abort();
                }
                
                //printf("factor: %f\n", factor);
                
                for (size_t subBlock = 0; subBlock < blockSize; subBlock++){
                    fPtr[blockID * blockSize + subBlock] = block.qs[subBlock] * factor;
                }
            }
        }else{
            better_assert(false, "Invalid blocked type in copy_to_onednn");
        }
    }
}