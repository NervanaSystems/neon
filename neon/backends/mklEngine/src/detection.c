//for ssd output detection fprop
//conf is a matrix with len*4*bs
//loc is a matrix with len*num_class*bs
//result is a matrix with bs
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mkl_trans.h>
#include <omp.h>

//self.bbox_transform_inv(prior_boxes, loc_view[:, :, k], self.proposals)
void bbox_transform_inv(
    float* boxes,       //boxes
    float* deltas,       //location
    float* output,       //output, num_boxes*4
    const long num_boxes)
{
    for(long i=0; i<num_boxes; ++i)
    {
        const float var0 = 0.1;
        const float var1 = 0.1;
        const float var2 = 0.2;
        const float var3 = 0.2;

        const long index = i * 4;
        const float widths  = boxes[index+2] - boxes[index];
        const float heights = boxes[index+3] - boxes[index+1];
        const float ctr_x = boxes[index]   + 0.5 * widths;
        const float ctr_y = boxes[index+1] + 0.5 * heights;

        const float dx = deltas[index];
        const float dy = deltas[index+1];
        const float dw = deltas[index+2];
        const float dh = deltas[index+3];

        const float pred_ctr_x = var0 * dx * widths + ctr_x;
        const float pred_ctr_y = var1 * dy * heights + ctr_y;
        const float pred_w = exp(var2 * dw) * widths;
        const float pred_h = exp(var3 * dh) * heights;

        output[index]   = pred_ctr_x - 0.5 * pred_w;
        output[index+1] = pred_ctr_y - 0.5 * pred_h;
        output[index+2] = pred_ctr_x + 0.5 * pred_w;
        output[index+3] = pred_ctr_y + 0.5 * pred_h;
    }
}


void softmax(float* input, int N, long long len)
{
    float* inPtr = (float*)input;
    for (int i = 0; i < N; ++i)
    {
        float *pTemp = inPtr + i * len;
        float pMax = pTemp[0];
        for(long long j = 0; j < len; ++j)
        {
            if (pMax < pTemp[j])
            {
                pMax = pTemp[j];
            }
        }
        float pSum = 0.0f;
        for(long long j=0; j<len; ++j)
        {
            pTemp[j] = exp(pTemp[j] - pMax);
            pSum += pTemp[j];
        }
        for(long long j=0; j < len; ++j)
        {
            pTemp[j] = pTemp[j] / pSum;
        }
    }
}

//find first N out of length
long get_top_N_index(
    float* scores,
    const long N,
    const long length,
    const float threshold,
    long* index)
{
    for(long i=0; i<length; ++i)
        index[i] = i;
    long num = (length<N) ? length:N;
    for(long i=0; i<num; ++i)
    {
        //find max score and store in index[i]
        for(long j=i+1; j<length; ++j)
        {
            if(scores[i]<scores[j])
            {
                float temp = scores[i];
                scores[i] = scores[j];
                scores[j] = temp;
                long temp_index = index[i];
                index[i] = index[j];
                index[j] = temp_index;
            }
        }

    }
    if(threshold>0)
    {
        long i = 0;
        for( ; i<num; ++i)
        {
            if(scores[i]<=threshold)
                break;
        }
        return i;
    }
    return num;
}


long nms(float* detection,
    long* index_sort,
    const float threshold,
    int normalized,
    const long N)
{
    float offset = (normalized) ? 0:1;
    int* out = malloc(N*sizeof(int));  //record swapped out choice
    float* area_vec = malloc(N*sizeof(float));
    for(long i=0; i<N; ++i)
        out[i] = 0;
    long i = 0;
    for(; i<N; ++i)
    {
        const float score = detection[i*5+4];
        if(score<=0)
            break;
        const float x1 = detection[i*5];
        const float y1 = detection[i*5+1];
        const float x2 = detection[i*5+2];
        const float y2 = detection[i*5+3];
        area_vec[i] = (x2 - x1 + offset) * (y2 - y1 + offset);
    }
    long positive_len = i;    //non-zero scores 0 ~ positive_len-1
    long result_len = 0;
    i = 0;
    while(i<positive_len)
    {
        if(out[i])   //this choice is kicked out
        {
            i++;
            continue;
        }
        index_sort[result_len] = i;
        result_len++;
        for(long j=i+1; j<positive_len; ++j)
        {
            if(out[j])
                continue;
            const float x11 = detection[i*5];
            const float y11 = detection[i*5+1];
            const float x12 = detection[i*5+2];
            const float y12 = detection[i*5+3];
            const float x21 = detection[j*5];
            const float y21 = detection[j*5+1];
            const float x22 = detection[j*5+2];
            const float y22 = detection[j*5+3];
            const float xx1 = (x11>x21) ? x11:x21;
            const float yy1 = (y11>y21) ? y11:y21;
            const float xx2 = (x12<x22) ? x12:x22;
            const float yy2 = (y12<y22) ? y12:y22;
            float w = xx2 - xx1 + offset;
            float h = yy2 - yy1 + offset;
            w = ( w > 0) ? w : 0;
            h = ( h > 0) ? h : 0;
            const float inter = w * h;
            const float ovr = inter / (area_vec[i] + area_vec[j] - inter);
            if (ovr > threshold)
                out[j] = 1;
        }
        i++;
    }
    free(out);
    free(area_vec);
    return result_len;
}


long detection_fprop(
    float* conf,          //score for each class for each box, num_box * num_class * bs
    float* loc,           //location for each box, box * 4 * bs
    float* res_detection, //final memory restoring boxes, bs * top_k
    float* prior_boxes,   //num_boxes * 4
    long * res_batch_len, //record count of result for each batch, bs
    const long num_boxes, //num_boxes, each is a potential object
    const long num_class, //number of class
    const long bs,        //batch size
    const long nms_topk,  //first top k box for nms result for each class
    const long image_topk,     //first top k box for input image
    const float score_threshold,  //threshold for accepting as a object for box
    const float nms_threshold)    //threshold for two overlapped boxes, too overlapped is one object
{
    //sorted result of index
    long* index_batch   = malloc(bs*num_boxes*num_class*sizeof(long));
    //scores to be sorted
    float* scores_batch = malloc(bs*num_boxes*num_class*sizeof(float));
    //temp result detections for each batch, grow when iterating among classes
    float* temp_res_detection_batch = malloc(bs*num_class*nms_topk*6*sizeof(float));
    //internal memory to restore sorted boxes for each class
    float* internal_detection_batch = malloc(bs*nms_topk*5*sizeof(float));
    //internal memory to restore transformed location
    float* proposal_batch = malloc(bs*num_boxes*4*sizeof(float));    

    //transpose KLN to NKL
    float* conf_t = malloc(num_boxes * num_class * bs * sizeof(float));
    float* loc_t  = malloc(num_boxes * 4* bs * sizeof(float));
    mkl_somatcopy('r', 't', num_boxes*num_class, bs, 1.0, conf, bs, conf_t, num_boxes*num_class);
    mkl_somatcopy('r', 't', num_boxes*4, bs, 1.0, loc, bs, loc_t, num_boxes*4);

    //loop for batch size
    #pragma omp parallel for
    for(long b=0; b<bs; ++b)  //loop for batch
    {
        float* scores = scores_batch + b * num_boxes*num_class;
        float* temp_res_detection = temp_res_detection_batch + b * num_class*nms_topk*6;
        long* index = index_batch + b * num_boxes*num_class;
        float* internal_detection = internal_detection_batch + b * nms_topk*5;
        float* proposal = proposal_batch + b * num_boxes*4;
        //calculate class scores for this batch using softmax
        float* conf_batch = conf_t + b * num_boxes * num_class;
        softmax(conf_batch, num_boxes, num_class);
        //store scores in an array
        mkl_somatcopy('r', 't', num_boxes, num_class, 1.0, conf_batch, num_class, scores, num_boxes);

        //transform locations in proposal
        bbox_transform_inv(prior_boxes, loc_t + b * 4 * num_boxes, proposal, num_boxes);

        long res_len = 0; //count of feasible boxes for this image
        for(long c=1; c<num_class; ++c) //loop for classes
        {
            //for each class, sort out first nms_topk boxes, store result in index
            long sort_nums_res = get_top_N_index(scores + c*num_boxes, nms_topk,
                           num_boxes, score_threshold, index);

            //store location and score for the sorted results
            if(sort_nums_res > 0)
            {
                //store location and score in internal_detection for overlapped check
                for(long i=0; i<sort_nums_res; ++i)
                {
                    for(long j=0; j<4; ++j)
                        internal_detection[i*5+j] = proposal[index[i]*4+j];
                    internal_detection[i*5+4] = scores[c*num_boxes+i];
                }

                //remove overlapped box
                sort_nums_res = nms(internal_detection, index, nms_threshold, 1, sort_nums_res);

                //store result in temp memory and add class number, thus width is 6
                for(long i=0; i<sort_nums_res; ++i)
                {
                    float* temp = temp_res_detection + (res_len+i)*6;
                    for(long j=0; j<5; ++j)
                    {
                        temp[j] = internal_detection[index[i]*5+j];
                    }
                    //add class number
                    temp[5] = c;
                }
                res_len += sort_nums_res;
            }
        }

        //sort out first top_k boxes for this image
        for(long i=0; i<res_len; ++i)
        {
            scores[i] = temp_res_detection[i*6+4];
            index[i] = i;
        }
        long sort_nums_res = res_len;
        if(sort_nums_res>image_topk) //sort first top_k out of res_len
        {
            sort_nums_res = get_top_N_index(scores, image_topk, res_len, 0.0, index);
        }

        //store sorted result in final output
        float* temp = res_detection + b * image_topk * 6;
        for(long i=0; i<sort_nums_res; ++i)
        {
            for(long j=0; j<6; ++j)
            {
                temp[i*6+j] = temp_res_detection[index[i]*6+j];
            }
        }
        res_batch_len[b] = sort_nums_res;
    }
    free(conf_t);
    free(loc_t);
    free(index_batch);
    free(scores_batch);
    free(temp_res_detection_batch);
    free(proposal_batch);
    free(internal_detection_batch);
}
