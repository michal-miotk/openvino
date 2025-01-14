// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if INPUT0_TYPE_SIZE == 2 //f16
#define HALF_ONE 0.5h
#else
#define HALF_ONE 0.5f
#endif

#define ZERO INPUT0_VAL_ZERO

#ifdef EDGPSI_STAGE_0

#define COORDINATES_OFFSET INPUT0_VAL_ONE

// 0. Refine anchors
KERNEL(edgpsi_ref_stage_0)
(const __global INPUT0_TYPE* im_info,
 const __global INPUT1_TYPE* anchors,
 const __global INPUT2_TYPE* deltas,
 const __global INPUT3_TYPE* scores,
 __global OUTPUT_TYPE* proposals) {
    const INPUT0_TYPE img_H = im_info[INPUT0_GET_INDEX(0, 0, 0, 0)];
    const INPUT0_TYPE img_W = im_info[INPUT0_GET_INDEX(1, 0, 0, 0)];

    const uint h = get_global_id(0);
    const uint w = get_global_id(1);
    const uint anchor = get_global_id(2);

    const uint anchor_idx = (h * BOTTOM_W + w) * ANCHORS_NUM + anchor;

    INPUT0_TYPE x0 = anchors[INPUT1_GET_INDEX(anchor_idx, 0, 0, 0)];
    INPUT0_TYPE y0 = anchors[INPUT1_GET_INDEX(anchor_idx, 1, 0, 0)];
    INPUT0_TYPE x1 = anchors[INPUT1_GET_INDEX(anchor_idx, 2, 0, 0)];
    INPUT0_TYPE y1 = anchors[INPUT1_GET_INDEX(anchor_idx, 3, 0, 0)];

    const INPUT0_TYPE dx = deltas[INPUT2_GET_INDEX(anchor * 4 + 0 , h, w, 0)];
    const INPUT0_TYPE dy = deltas[INPUT2_GET_INDEX(anchor * 4 + 1 , h , w, 0)];
    const INPUT0_TYPE d_log_w = deltas[INPUT2_GET_INDEX(anchor * 4 + 2 , h, w, 0)];
    const INPUT0_TYPE d_log_h = deltas[INPUT2_GET_INDEX(anchor * 4 + 3 , h, w, 0)];

    const INPUT0_TYPE score = scores[INPUT3_GET_INDEX(anchor, h, w, 0)];

    // width & height of box
    const INPUT0_TYPE ww = x1 - x0 + COORDINATES_OFFSET;
    const INPUT0_TYPE hh = y1 - y0 + COORDINATES_OFFSET;
    // center location of box
    const INPUT0_TYPE ctr_x = x0 + HALF_ONE * ww;
    const INPUT0_TYPE ctr_y = y0 + HALF_ONE * hh;

    // new center location according to deltas (dx, dy)
    const INPUT0_TYPE pred_ctr_x = dx * ww + ctr_x;
    const INPUT0_TYPE pred_ctr_y = dy * hh + ctr_y;
    // new width & height according to deltas d(log w), d(log h)
    const INPUT0_TYPE pred_w = exp(min(d_log_w, TO_INPUT0_TYPE(MAX_DELTA_LOG_WH))) * ww;
    const INPUT0_TYPE pred_h = exp(min(d_log_h, TO_INPUT0_TYPE(MAX_DELTA_LOG_WH))) * hh;

    // update upper-left corner location
    x0 = pred_ctr_x - HALF_ONE * pred_w;
    y0 = pred_ctr_y - HALF_ONE * pred_h;
    // update lower-right corner location
    x1 = pred_ctr_x + HALF_ONE * pred_w - COORDINATES_OFFSET;
    y1 = pred_ctr_y + HALF_ONE * pred_h - COORDINATES_OFFSET;

    // adjust new corner locations to be within the image region
    x0 = max(ZERO, min(x0, img_W - COORDINATES_OFFSET));
    y0 = max(ZERO, min(y0, img_H - COORDINATES_OFFSET));
    x1 = max(ZERO, min(x1, img_W - COORDINATES_OFFSET));
    y1 = max(ZERO, min(y1, img_H - COORDINATES_OFFSET));

    // recompute new width & height
    const INPUT0_TYPE box_w = x1 - x0 + COORDINATES_OFFSET;
    const INPUT0_TYPE box_h = y1 - y0 + COORDINATES_OFFSET;

    const uint proposal_idx = anchor_idx * 5;
    proposals[proposal_idx + 0] = x0;
    proposals[proposal_idx + 1] = y0;
    proposals[proposal_idx + 2] = x1;
    proposals[proposal_idx + 3] = y1;
    proposals[proposal_idx + 4] = ((MIN_SIZE <= box_w) && (MIN_SIZE <= box_h)) ? score : 0.f;
}

#undef COORDINATES_OFFSET

#endif /* EDGPSI_STAGE_0 */

#ifdef EDGPSI_STAGE_1
#define Box FUNC(_Box)
typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE x0;
    INPUT0_TYPE y0;
    INPUT0_TYPE x1;
    INPUT0_TYPE y1;
    INPUT0_TYPE score;
} Box;

#define heapsizes FUNC(_heapsizes)
typedef struct __attribute__((__packed__))
{
    uint mask; // Leo. nums. in use (sizes of existing heaps)
    uint offset;    // Add this to every bit's position ('mask'
} heapsizes; 

inline void FUNC(swap_box)(__global Box* a, __global Box* b) {
    const Box temp = *a;
    *a = *b;
    *b = temp;
}

inline void FUNC(sift_in)(Box* root, uint size)
{
    static const uint L[46] =     // Leonardo numbers in [0,1<<32)
    {
        1UL, 1UL, 3UL, 5UL, 9UL, 15UL, 25UL, 41UL, 67UL, 109UL, 177UL,
        287UL, 465UL, 753UL, 1219UL, 1973UL, 3193UL, 5167UL, 8361UL,
        13529UL, 21891UL, 35421UL, 57313UL, 92735UL, 150049UL,
        242785UL, 392835UL, 635621UL, 1028457UL, 1664079UL, 2692537UL,
        4356617UL, 7049155UL, 11405773UL, 18454929UL, 29860703UL,
        48315633UL, 78176337UL, 126491971UL, 204668309UL, 331160281UL,
        535828591UL, 866988873UL, 1402817465UL, 2269806339UL,
        3672623805UL
    };
    Box * left;
    Box* right; // Pos. of children heaps
    Box * next;          // Chosen child (greater root)
    Box tmp;             // Value to move down
    uint nsz;                        // Size of chosen child heap
    
    if (size < 2)        // If we are in a leaf,
        return;          // there's nothing to do

    tmp = *root;         // Backup the initial value
    
    do                        // While there are children heaps...
    {
        right = root - 1;           // Locate children
        left = right - L[size-2];
        
        if (right->score < left->score)         // Compare their roots
        {
            next = left;            // Choose left child heap
            nsz = size - 1;         // (larger subheap)
        }
        else
        {
            next = right;           // Choose right child heap
            nsz = size - 2;         // (smaller subheap)
        }
                                    // If both roots are less than
        if (next->score <= tmp.score)           // the initial root, we have
            break;                  // reached its final position

        *root = *next;              // Otherwise, push up the
                                    // greater root and
        root = next;                // proceed down to the
        size = nsz;                 // next level
    }
    while (size > 1);          // If we reach a leaf, stop
    
    *root = tmp;         // Write the initial value in its
}                        // final position

inline void FUNC(interheap_sift)(Box* root, heapsizes hsz)
{
    static const uint L[46] =     // Leonardo numbers in [0,1<<32)
    {
        1UL, 1UL, 3UL, 5UL, 9UL, 15UL, 25UL, 41UL, 67UL, 109UL, 177UL,
        287UL, 465UL, 753UL, 1219UL, 1973UL, 3193UL, 5167UL, 8361UL,
        13529UL, 21891UL, 35421UL, 57313UL, 92735UL, 150049UL,
        242785UL, 392835UL, 635621UL, 1028457UL, 1664079UL, 2692537UL,
        4356617UL, 7049155UL, 11405773UL, 18454929UL, 29860703UL,
        48315633UL, 78176337UL, 126491971UL, 204668309UL, 331160281UL,
        535828591UL, 866988873UL, 1402817465UL, 2269806339UL,
        3672623805UL
    };
    Box * next;   // Pos. of (root of) next heap
    Box * left;   // Pos. of left child of current heap
    Box * right;  //  "   "  right  "   "     "     "
    Box tmp;      // Value to move left
    Box max;      // Effective root value of curr. heap
    
    tmp = *root;      // Backup the initial value
    
    while (hsz.mask != 1)  // Traverse the list of heaps
    {                      // from right to left
        max = tmp;
        
        if (hsz.offset > 1)           // If this heap has children
        {
            right = root - 1;                 // Locate children
            left = right - L[hsz.offset-2];
            
            if (max.score < left->score)                  // Use the maximum
                max = *left;                  // value for the
                                              // comparison below,
            if (max.score < right->score)                 // since it is the
                max = *right;                 // effective root
        }                                     // of this heap
        
        next = root - L[hsz.offset];  // Position of next heap

        if (next->score <= max.score)             // If the ordeing is OK,
            break;                    // stop here

        *root = *next;                // Otherwise, push up the
        root = next;                  // root of that heap and
                                      // go there
        do
        {                             // Extract the previous
            hsz.mask >>= 1;           // heap from the list (note
            hsz.offset ++;            // that 'hsz' is just a
        }                             // temporary copy)
        while (!(hsz.mask&1));
    }
                                      // Put the initial root in
    *root = tmp;                      // the heap where we stopped
    FUNC_CALL(sift_in)(root, hsz.offset);       // and ensure the correct
}

inline void FUNC(extract)(__global Box* A, uint num, heapsizes hsz)
{
    static const uint L[46] =     // Leonardo numbers in [0,1<<32)
    {
        1UL, 1UL, 3UL, 5UL, 9UL, 15UL, 25UL, 41UL, 67UL, 109UL, 177UL,
        287UL, 465UL, 753UL, 1219UL, 1973UL, 3193UL, 5167UL, 8361UL,
        13529UL, 21891UL, 35421UL, 57313UL, 92735UL, 150049UL,
        242785UL, 392835UL, 635621UL, 1028457UL, 1664079UL, 2692537UL,
        4356617UL, 7049155UL, 11405773UL, 18454929UL, 29860703UL,
        48315633UL, 78176337UL, 126491971UL, 204668309UL, 331160281UL,
        535828591UL, 866988873UL, 1402817465UL, 2269806339UL,
        3672623805UL
    };
    uint i;          // Loop index for traversing the array

    uint ch[2];      // Position of left and right children
                         // of a newly created heap
    uint j;
                             // Extract elems. starting at the end
    for (i=num-1; i>1; i--)  // When only two remain, it's done
    {
        if (hsz.offset<2)         // If last heap has size L[1] or
        {                         // L[0] (both ==1), just remove
            do                    // this heap (update the
            {                     // heapsizes struct) leaving the
                hsz.mask >>= 1;   // single element untouched
                hsz.offset ++;
            }                       // The mask will never be 0
            while (!(hsz.mask&1));  // because the loop terminates
        }                           // early (with two heaps of
        else                        // sizes L[1] and L[0])
        {
            ch[1] = i - 1;                   // Position of right
            ch[0] = ch[1] - L[hsz.offset-2]; // and left children

            hsz.mask &= ~1;       // Remove current heap

            for (j=0; j<2; j++)      // For every child heap (left
            {                                            // first)
                hsz.mask = (hsz.mask << 1) | 1;
                hsz.offset --;                  // Add heap to the
                                                // list and ensure
                FUNC_CALL(interheap_sift)(A+ch[j], hsz);  // ordering of
            }                                   // roots
        }
    }
}
inline heapsizes FUNC(heapify)(__global Box* A, uint num)
{
    static const uint L[46] =     // Leonardo numbers in [0,1<<32)
    {
        1UL, 1UL, 3UL, 5UL, 9UL, 15UL, 25UL, 41UL, 67UL, 109UL, 177UL,
        287UL, 465UL, 753UL, 1219UL, 1973UL, 3193UL, 5167UL, 8361UL,
        13529UL, 21891UL, 35421UL, 57313UL, 92735UL, 150049UL,
        242785UL, 392835UL, 635621UL, 1028457UL, 1664079UL, 2692537UL,
        4356617UL, 7049155UL, 11405773UL, 18454929UL, 29860703UL,
        48315633UL, 78176337UL, 126491971UL, 204668309UL, 331160281UL,
        535828591UL, 866988873UL, 1402817465UL, 2269806339UL,
        3672623805UL
    };
    heapsizes hsz;       // 'List' of sizes of existing heaps
    
    uint i;          // Loop index for traversing the array
    
    uint wbf;             // Flag indicating whether a newly
                         // created heap will be fused later in a
                         // larger heap (wbf!=0) or not (wbf==0)

    hsz.mask = 1;             // Create a heap of size L[1]
    hsz.offset = 1;           // containing the first element

    for (i=1; i<num; i++)     // With every following element...
    {
        if (hsz.mask & 2)          // If possible (if contiguous
        {                                  // Leonardo numbers),
            hsz.mask = (hsz.mask>>2) | 1;  // fuse last two heaps
            hsz.offset += 2;
        }                          // Otherwise,
        else if (hsz.offset == 1)  // if last heap has size L[1]
        {
            hsz.mask = (hsz.mask << 1) | 1;  // the new is L[0]
            hsz.offset = 0;
        }
        else                       // Otherwise, new heap L[1]
        {
            hsz.mask = (hsz.mask << (hsz.offset-1)) | 1;
            hsz.offset = 1;
        }
        
            // The current heap will be fused in the future if:
            //
            //     a) The sizes of this heap and the previous are
            //        contiguous Leonardo numbers AND there is at
            //        least one more element in the array
            //  OR
            //     b) This heap has size L[x] where x>0 AND there
            //        is still space for a heap of size L[x-1] and
            //        one more element (L[x]+L[x-1]+1 --> L[x+1])

        wbf = ( (hsz.mask & 2) &&
                i+1 < num                 ) ||
              ( hsz.offset > 0    &&
                (uint)(1)+i+L[hsz.offset-1] < num );

        if (wbf)                       // If this new heap will be
            FUNC_CALL(sift_in)(A+i, hsz.offset); // fused, don't propagate
        else                           // the root (just fix this
            FUNC_CALL(interheap_sift)(A+i, hsz); // heap). If it will _not_
    }                                  // be fused, propagate the
                                       // root through the
    return hsz;                        // sequence of heaps to
}

inline void FUNC(smoothsort)(__global Box* A, uint num)
{
    heapsizes hsz;
 
    if (num < 2)  // If there's only one element, it's done.
        return;   // The other functions assume 2 or more elements

    hsz = FUNC_CALL(heapify)(A, num);   // Build the ordered list of heaps

    FUNC_CALL(extract)(A, num, hsz);    // Consume the list of heaps
}
// 1. Sort boxes by scores
KERNEL(edgpsi_ref_stage_1)(__global OUTPUT_TYPE* proposals) {
    __global Box* boxes = (__global Box*)proposals;
    static const uint L[46] =     // Leonardo numbers in [0,1<<32)
    {
        1UL, 1UL, 3UL, 5UL, 9UL, 15UL, 25UL, 41UL, 67UL, 109UL, 177UL,
        287UL, 465UL, 753UL, 1219UL, 1973UL, 3193UL, 5167UL, 8361UL,
        13529UL, 21891UL, 35421UL, 57313UL, 92735UL, 150049UL,
        242785UL, 392835UL, 635621UL, 1028457UL, 1664079UL, 2692537UL,
        4356617UL, 7049155UL, 11405773UL, 18454929UL, 29860703UL,
        48315633UL, 78176337UL, 126491971UL, 204668309UL, 331160281UL,
        535828591UL, 866988873UL, 1402817465UL, 2269806339UL,
        3672623805UL
    };
    FUNC_CALL(smoothsort)(boxes, NUM_PROPOSALS);
}
#undef Box
#endif /* EDGPSI_STAGE_1 */

#ifdef EDGPSI_STAGE_2

// 2. NMS
KERNEL(edgpsi_ref_stage_2)
(const __global INPUT0_TYPE* boxes, __global size_t* out_indices, __global size_t* num_outputs) {
    uint count = 0;
    uint index_out[POST_NMS_COUNT] = {0};

    uint is_dead[PRE_NMS_TOPN] = {0};
    for (uint box = 0; box < PRE_NMS_TOPN; ++box) {
        if (is_dead[box])
            continue;

        index_out[count++] = box;
        if (count == POST_NMS_COUNT)
            break;

        const uint box_offset = box * 5;
        const INPUT0_TYPE x0i = boxes[box_offset + 0];
        const INPUT0_TYPE y0i = boxes[box_offset + 1];
        const INPUT0_TYPE x1i = boxes[box_offset + 2];
        const INPUT0_TYPE y1i = boxes[box_offset + 3];

        const INPUT0_TYPE a_width = x1i - x0i;
        const INPUT0_TYPE a_height = y1i - y0i;
        const INPUT0_TYPE a_area = a_width * a_height;

        for (uint tail = box + 1; tail < PRE_NMS_TOPN; ++tail) {
            const uint tail_offset = tail * 5;
            const INPUT0_TYPE x0j = boxes[tail_offset + 0];
            const INPUT0_TYPE y0j = boxes[tail_offset + 1];
            const INPUT0_TYPE x1j = boxes[tail_offset + 2];
            const INPUT0_TYPE y1j = boxes[tail_offset + 3];

            const INPUT0_TYPE x0 = max(x0i, x0j);
            const INPUT0_TYPE y0 = max(y0i, y0j);
            const INPUT0_TYPE x1 = min(x1i, x1j);
            const INPUT0_TYPE y1 = min(y1i, y1j);

            const INPUT0_TYPE width = x1 - x0;
            const INPUT0_TYPE height = y1 - y0;
            const INPUT0_TYPE area = max(ZERO, width) * max(ZERO, height);

            const INPUT0_TYPE b_width = x1j - x0j;
            const INPUT0_TYPE b_height = y1j - y0j;
            const INPUT0_TYPE b_area = b_width * b_height;

            const INPUT0_TYPE intersection_area = area / (a_area + b_area - area);

            is_dead[tail] =
                (NMS_THRESHOLD < intersection_area) && (x0i <= x1j) && (y0i <= y1j) && (x0j <= x1i) && (y0j <= y1i);
        }
    }

    *num_outputs = count;
    for (uint i = 0; i < count; ++i) {
        out_indices[i] = index_out[i];
    }
}
#endif /* EDGPSI_STAGE_2 */

#ifdef EDGPSI_STAGE_3

// 3. Convert proposals to rois and roi_scores
KERNEL(edgpsi_ref_stage_3)
(const __global INPUT0_TYPE* boxes,
 const __global size_t* out_indices,
 const __global size_t* num_outputs,
 __global OUTPUT_TYPE* rois,
 __global OUTPUT_TYPE* roi_scores) {
    const uint i = get_global_id(0);
    const uint index = out_indices[i];
    const uint box_offset = index * 5;
    const uint rois_offset = i * 4;

    if (i < *num_outputs) {
        rois[OUTPUT_GET_INDEX(i, 0, 0, 0)] = boxes[box_offset + 0];
        rois[OUTPUT_GET_INDEX(i, 1, 0, 0)] = boxes[box_offset + 1];
        rois[OUTPUT_GET_INDEX(i, 2, 0, 0)] = boxes[box_offset + 2];
        rois[OUTPUT_GET_INDEX(i, 3, 0, 0)] = boxes[box_offset + 3];
        roi_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = boxes[box_offset + 4];
    } else {
        rois[OUTPUT_GET_INDEX(i, 0, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 1, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 2, 0, 0)] = 0.0f;
        rois[OUTPUT_GET_INDEX(i, 3, 0, 0)] = 0.0f;
        roi_scores[OUTPUT1_GET_INDEX(i, 0, 0, 0)] = 0.0f;
    }
}
#endif /* EDGPSI_STAGE_3 */

#undef HALF_ONE
