/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "fastcv_hal_imgproc.hpp"
#include "fastcv_hal_utils.hpp"
#include <opencv2/core/base.hpp>
#include <opencv2/imgproc.hpp>


int fastcv_hal_medianBlur(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             depth,
    int             cn,
    int             ksize)
{
    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input image width and height should greater than kernel size
    if ((height <= ksize) || (width <= ksize))
        CV_HAL_RETURN_NOT_IMPLEMENTED("Input image size should be larger than kernel size");

    // The input channel should be 1
    if (cn != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    INITIALIZATION_CHECK;

    fcvStatus status = FASTCV_SUCCESS;
    int fcvFuncType = FCV_MAKETYPE(ksize,depth);

    switch (fcvFuncType)
    {
        case FCV_MAKETYPE(3,CV_8U):
        {
            status = fcvFilterMedian3x3u8_v3(src_data, width, height, src_step, dst_data, dst_step,
                fcvBorderType::FASTCV_BORDER_REPLICATE, 0);
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, depth:%s is not supported", ksize, cv::depthToString(depth)));
    }

    CV_HAL_RETURN(status, hal_medianBlur);
}

class FcvSobelLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvSobelLoop_Invoker(const cv::Mat& _src, cv::Mat& _dst, int _dx, int _dy, int _ksize, fcvBorderType _fcvBorder,
        int _fcvBorderValue) : cv::ParallelLoopBody(), src(_src), dst(_dst), dx(_dx), dy(_dy), ksize(_ksize),
        fcvBorder(_fcvBorder), fcvBorderValue(_fcvBorderValue)
    {
        width       = src.cols;
        height      = src.rows;
        halfKernelSize  = ksize/2;
        fcvFuncType = FCV_MAKETYPE(ksize,src.depth());
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        // Need additional lines to be border.
        if(range.start > 0)
        {
            topLines     += halfKernelSize;
            paddedHeight += halfKernelSize;
        }

        if(range.end < height)
        {
            paddedHeight += halfKernelSize;
        }

        cv::Mat srcPadded = src(cv::Rect(0, range.start-topLines, width, paddedHeight));
        cv::Mat dstPadded = cv::Mat(paddedHeight, width, dst.depth());

        int16_t *dxBuffer = nullptr, *dyBuffer = nullptr;

        if ((dx == 1) && (dy == 0))
        {
            dxBuffer = (int16_t*)dstPadded.data;
        }
        else if ((dx == 0) && (dy == 1))
        {
            dyBuffer = (int16_t*)dstPadded.data;
        }

        switch (fcvFuncType)
        {
            case FCV_MAKETYPE(3,CV_8U):
            {
                fcvFilterSobel3x3u8s16(srcPadded.data, width, paddedHeight, srcPadded.step, dxBuffer, dyBuffer, dstPadded.step,
                    fcvBorder, 0);
                break;
            }
            case FCV_MAKETYPE(5,CV_8U):
            {
                fcvFilterSobel5x5u8s16(srcPadded.data, width, paddedHeight, srcPadded.step, dxBuffer, dyBuffer, dstPadded.step,
                    fcvBorder, 0);
                break;
            }
            case FCV_MAKETYPE(7,CV_8U):
            {
                fcvFilterSobel7x7u8s16(srcPadded.data, width, paddedHeight, srcPadded.step, dxBuffer, dyBuffer, dstPadded.step,
                    fcvBorder, 0);
                break;
            }
            default:
                CV_Error(cv::Error::StsBadArg, cv::format("Ksize:%d, src_depth:%s is not supported",
                    ksize, cv::depthToString(src.depth())));
                break;
        }

        // Only copy center part back to output image and ignore the padded lines
        cv::Mat temp1 = dstPadded(cv::Rect(0, topLines, width, rangeHeight));
        cv::Mat temp2 = dst(cv::Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const cv::Mat&  src;
    cv::Mat&        dst;
    int             width;
    int             height;
    int             dx;
    int             dy;
    int             ksize;
    int             halfKernelSize;
    int             fcvFuncType;
    fcvBorderType   fcvBorder;
    int             fcvBorderValue;

    FcvSobelLoop_Invoker(const FcvSobelLoop_Invoker &);  // = delete;
    const FcvSobelLoop_Invoker& operator= (const FcvSobelLoop_Invoker &);  // = delete;
};

int fastcv_hal_sobel(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             src_depth,
    int             dst_depth,
    int             cn,
    int             margin_left,
    int             margin_top,
    int             margin_right,
    int             margin_bottom,
    int             dx,
    int             dy,
    int             ksize,
    double          scale,
    double          delta,
    int             border_type)
{
    if (!(FCV_CMP_EQ(scale, 1.0f) && FCV_CMP_EQ(delta, 0.0f)))
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Scale:%f, delta:%f is not supported", scale, delta));

    // Only support one direction derivatives and the order is 1.(dx=1 && dy=0)||(dx=0 && dy=1)
    if ((dx + dy == 0) || (dx + dy > 1))
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Dx:%d Dy:%d is not supported",dx, dy));

    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input image width and height should greater than kernel size
    if ((height <= ksize) || (width <= ksize))
        CV_HAL_RETURN_NOT_IMPLEMENTED("Input image size should be larger than kernel size");

    // The input channel should be 1
    if (cn != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    // 1. When ksize <= 0, OpenCV will use Scharr Derivatives
    // 2. When ksize == 1, OpenCV will use 3×1 or 1×3 kernel(no Gaussian smoothing is done)
    // FastCV doesn't support above two situation
    if (ksize <= 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Scharr derivatives or non square kernel are not supported");

    // Only support the result type is CV_16S
    if (dst_depth != CV_16S)
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Dst depth:%s is not supported", cv::depthToString(dst_depth)));

    // Only support following ksize and src_depth as input
    if ((FCV_MAKETYPE(ksize,src_depth) != FCV_MAKETYPE(3, CV_8U))   &&
        (FCV_MAKETYPE(ksize,src_depth) != FCV_MAKETYPE(5, CV_8U))   &&
        (FCV_MAKETYPE(ksize,src_depth) != FCV_MAKETYPE(7, CV_8U)))
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, src_depth:%s is not supported", ksize, cv::depthToString(src_depth)));

    INITIALIZATION_CHECK;

    fcvStatus       status    = FASTCV_SUCCESS;
    fcvBorderType   fcvBorder = FASTCV_BORDER_CONSTANT;

    switch (border_type)
    {
        // For constant border, there are no border value, OpenCV default value is 0
        case cv::BorderTypes::BORDER_CONSTANT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_CONSTANT;
            break;
        }
        case cv::BorderTypes::BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Border type:%s is not supported", borderToString(border_type)));
    }

    cv::Mat src = cv::Mat(height, width, CV_MAKE_TYPE(src_depth, 1), (void*)src_data, src_step);
    cv::Mat dst = cv::Mat(height, width, CV_MAKE_TYPE(dst_depth, 1), (void*)dst_data, dst_step);

    if (margin_left||margin_top||margin_top||margin_bottom)
    {
        // Need additional lines to be border.
        int paddedHeight = height, paddedWidth = width, startX = 0, startY = 0;

        if(margin_left != 0)
        {
            src_data    -= ksize/2;
            paddedWidth += ksize/2;
            startX      =  ksize/2;
        }

        if(margin_top != 0)
        {
            src_data     -= (ksize/2) * src_step;
            paddedHeight += ksize/2;
            startY       =  ksize/2;
        }

        if(margin_right != 0)
        {
            paddedWidth += ksize/2;
        }

        if(margin_bottom != 0)
        {
            paddedHeight += ksize/2;
        }

        cv::Mat padded(paddedHeight, paddedWidth, src_depth);
        int16_t *dxBuffer = nullptr, *dyBuffer = nullptr;

        if ((dx == 1) && (dy == 0))
        {
            dxBuffer = (int16_t*)padded.data;
            dyBuffer = NULL;
        }
        else if ((dx == 0) && (dy == 1))
        {
            dxBuffer = NULL;
            dyBuffer = (int16_t*)padded.data;
        }

        int fcvFuncType = FCV_MAKETYPE(ksize, src_depth);

        switch (fcvFuncType)
        {
            case FCV_MAKETYPE(3,CV_8U):
            {
                status = fcvFilterSobel3x3u8s16(src_data, paddedWidth, paddedHeight, src_step, dxBuffer, dyBuffer, padded.step,
                    fcvBorder, 0);
                break;
            }
            case FCV_MAKETYPE(5,CV_8U):
            {
                status = fcvFilterSobel5x5u8s16(src_data, paddedWidth, paddedHeight, src_step, dxBuffer, dyBuffer, padded.step,
                    fcvBorder, 0);
                break;
            }
            case FCV_MAKETYPE(7,CV_8U):
            {
                status = fcvFilterSobel7x7u8s16(src_data, paddedWidth, paddedHeight, src_step, dxBuffer, dyBuffer, padded.step,
                    fcvBorder, 0);
                break;
            }
            default:
                CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, src_depth:%s is not supported",
                    ksize, cv::depthToString(src_depth)));
                break;
        }

        cv::Mat temp1 = padded(cv::Rect(startX, startY, width, height));
        temp1.copyTo(dst);
    }
    else
    {
        int nThreads = cv::getNumThreads();
        int nStripes = nThreads > 1 ? 3*nThreads : 1;

        cv::parallel_for_(cv::Range(0, height), FcvSobelLoop_Invoker(src, dst, dx, dy, ksize, fcvBorder, 0), nStripes);
    }

    CV_HAL_RETURN(status, hal_sobel);
}

int fastcv_hal_boxFilter(
    const uchar*     src_data,
    size_t           src_step,
    uchar*           dst_data,
    size_t           dst_step,
    int              width,
    int              height,
    int              src_depth,
    int              dst_depth,
    int              cn,
    int              margin_left,
    int              margin_top,
    int              margin_right,
    int              margin_bottom,
    size_t           ksize_width,
    size_t           ksize_height,
    int              anchor_x,
    int              anchor_y,
    bool             normalize,
    int              border_type)
{
    if((width*height) < (320*240))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("input size not supported");
    }
    else if(src_data == dst_data)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("in-place processing not supported");
    }
    else if(src_depth != CV_8U || cn != 1)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("src type not supported");
    }
    else if(dst_depth != src_depth)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("same src and dst type supported");
    }
    else if(ksize_width != ksize_height ||
           (ksize_width != 3 && ksize_width != 5))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("kernel size not supported");
    }
    else if(anchor_x != -1 || anchor_y != -1 ||
            margin_left != 0 || margin_top != 0 ||
            margin_right != 0 || margin_bottom != 0)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("ROI not supported");
    }

    INITIALIZATION_CHECK;

    fcvBorderType bdr;
    uint8_t bdrVal = 0;
    switch(border_type)
    {
        case cv::BORDER_REPLICATE:
            bdr = FASTCV_BORDER_REPLICATE;
            break;
        case cv::BORDER_REFLECT:
            bdr = FASTCV_BORDER_REFLECT;
            break;
        case cv::BORDER_REFLECT101:    // cv::BORDER_REFLECT_101, BORDER_DEFAULT
            bdr = FASTCV_BORDER_REFLECT_V2;
            break;
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED("border type not supported");
    }

    fcvStatus status = FASTCV_SUCCESS;
    if(ksize_width == 3)
    {
        status = fcvBoxFilter3x3u8_v3(src_data, width, height, src_step,
                                      dst_data, dst_step, normalize, bdr, bdrVal);
    }
    else if(ksize_width == 5)
    {
        status = fcvBoxFilter5x5u8_v2(src_data, width, height, src_step,
                                      dst_data, dst_step, normalize, bdr, bdrVal);
    }

    CV_HAL_RETURN(status,hal_boxFilter);
}

int fastcv_hal_adaptiveThreshold(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    double          maxValue,
    int             adaptiveMethod,
    int             thresholdType,
    int             blockSize,
    double          C)
{
    if((width*height) < (320*240))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("input size not supported");
    }
    else if (src_data == dst_data)
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("In place processing not supported");
    }

    int value  = (thresholdType == cv::THRESH_BINARY) ? cvCeil(C) : cvFloor(C);

    if ((maxValue < 1) || (maxValue > 255))
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("max value 1-255 supported");
    }

    INITIALIZATION_CHECK;

    uchar maxVal  = cv::saturate_cast<uchar>(maxValue);

    fcvThreshType threshType = (thresholdType == cv::THRESH_BINARY) ? FCV_THRESH_BINARY : FCV_THRESH_BINARY_INV;

    fcvStatus status = FASTCV_SUCCESS;
    if(adaptiveMethod == cv::ADAPTIVE_THRESH_GAUSSIAN_C)
    {
        if(blockSize == 3)
            status = fcvAdaptiveThresholdGaussian3x3u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else if(blockSize == 5)
            status = fcvAdaptiveThresholdGaussian5x5u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else
        {
            CV_HAL_RETURN_NOT_IMPLEMENTED("block size not supported");
        }
    }
    else if(adaptiveMethod == cv::ADAPTIVE_THRESH_MEAN_C)
    {
        if(blockSize == 3)
            status = fcvAdaptiveThresholdMean3x3u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else if(blockSize == 5)
            status = fcvAdaptiveThresholdMean5x5u8_v2(src_data, width, height, src_step, maxVal, threshType, value, dst_data, dst_step);
        else
        {
            CV_HAL_RETURN_NOT_IMPLEMENTED("block size not supported");
        }
    }
    else
    {
        CV_HAL_RETURN_NOT_IMPLEMENTED("adaptive method not supported");
    }

    CV_HAL_RETURN(status,hal_adaptiveThreshold);
}

class FcvGaussianBlurLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvGaussianBlurLoop_Invoker(const cv::Mat& _src, cv::Mat& _dst, int _ksize, fcvBorderType _fcvBorder, int _fcvBorderValue) :
        cv::ParallelLoopBody(), src(_src),dst(_dst), ksize(_ksize), fcvBorder(_fcvBorder), fcvBorderValue(_fcvBorderValue)
    {
        width       = src.cols;
        height      = src.rows;
        halfKernelSize   = ksize / 2;
        fcvFuncType = FCV_MAKETYPE(ksize, src.depth());
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        int topLines     = 0;
        int rangeHeight  = range.end-range.start;
        int paddedHeight = rangeHeight;

        // Need additional lines to be border.
        if(range.start != 0)
        {
            topLines     += halfKernelSize;
            paddedHeight += halfKernelSize;
        }

        if(range.end != height)
        {
            paddedHeight += halfKernelSize;
        }

        const cv::Mat srcPadded = src(cv::Rect(0, range.start - topLines, width, paddedHeight));
        cv::Mat dstPadded       = cv::Mat(paddedHeight, width, CV_8U);

        if (fcvFuncType == FCV_MAKETYPE(3,CV_8U))
            fcvFilterGaussian3x3u8_v4(srcPadded.data, width, paddedHeight, srcPadded.step, dstPadded.data, dstPadded.step,
                fcvBorder, 0);
        else if (fcvFuncType == FCV_MAKETYPE(5,CV_8U))
            fcvFilterGaussian5x5u8_v3(srcPadded.data, width, paddedHeight, srcPadded.step, dstPadded.data, dstPadded.step,
                fcvBorder, 0);

        // Only copy center part back to output image and ignore the padded lines
        cv::Mat temp1 = dstPadded(cv::Rect(0, topLines, width, rangeHeight));
        cv::Mat temp2 = dst(cv::Rect(0, range.start, width, rangeHeight));
        temp1.copyTo(temp2);
    }

    private:
    const cv::Mat&  src;
    cv::Mat&        dst;
    int             width;
    int             height;
    const int       ksize;
    int             halfKernelSize;
    int             fcvFuncType;
    fcvBorderType   fcvBorder;
    int             fcvBorderValue;

    FcvGaussianBlurLoop_Invoker(const FcvGaussianBlurLoop_Invoker &);  // = delete;
    const FcvGaussianBlurLoop_Invoker& operator= (const FcvGaussianBlurLoop_Invoker &);  // = delete;
};

int fastcv_hal_gaussianBlurBinomial(
    const uchar*    src_data,
    size_t          src_step,
    uchar*          dst_data,
    size_t          dst_step,
    int             width,
    int             height,
    int             depth,
    int             cn,
    size_t          margin_left,
    size_t          margin_top,
    size_t          margin_right,
    size_t          margin_bottom,
    size_t          ksize,
    int             border_type)
{
    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input image width and height should greater than kernel size
    if (((size_t)height <= ksize) || ((size_t)width <= ksize))
        CV_HAL_RETURN_NOT_IMPLEMENTED("Input image size should be larger than kernel size");

    // The input channel should be 1
    if (cn != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    // Do not support for ROI case
    if((margin_left!=0) || (margin_top != 0) || (margin_right != 0) || (margin_bottom !=0))
        CV_HAL_RETURN_NOT_IMPLEMENTED("ROI is not supported");

    INITIALIZATION_CHECK;

    fcvStatus status = FASTCV_SUCCESS;
    fcvBorderType fcvBorder = fcvBorderType::FASTCV_BORDER_UNDEFINED;
    int fcvFuncType = FCV_MAKETYPE(ksize,depth);

    switch (border_type)
    {
        case cv::BorderTypes::BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        // For constant border, there are no border value, OpenCV default value is 0
        case cv::BorderTypes::BORDER_CONSTANT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_CONSTANT;
            break;
        }
        case cv::BorderTypes::BORDER_REFLECT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REFLECT;
            break;
        }
        case cv::BorderTypes::BORDER_REFLECT_101:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REFLECT_V2;
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Border type:%s is not supported", borderToString(border_type)));
    }

    int nThreads = cv::getNumThreads();
    int nStripes = (nThreads > 1) ? ((height > 60) ? 3 * nThreads : 1) : 1;

    switch (fcvFuncType)
    {
        case FCV_MAKETYPE(3,CV_8U):
        case FCV_MAKETYPE(5,CV_8U):
        {
            cv::Mat src = cv::Mat(height, width, CV_8UC1, (void*)src_data, src_step);
            cv::Mat dst = cv::Mat(height, width, CV_8UC1, (void*)dst_data, dst_step);
            cv::parallel_for_(cv::Range(0, height), FcvGaussianBlurLoop_Invoker(src, dst, ksize, fcvBorder, 0), nStripes);
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Ksize:%d, depth:%s is not supported", (int)ksize, cv::depthToString(depth)));
    }

    CV_HAL_RETURN(status, hal_gaussianBlurBinomial);
}

class FcvWarpPerspectiveLoop_Invoker : public cv::ParallelLoopBody
{
    public:

    FcvWarpPerspectiveLoop_Invoker(const uchar* _src_data, int _src_width, int _src_height, size_t _src_step, uchar* _dst_data,
        int _dst_width, int _dst_height, size_t _dst_step, int _type, const double* _M,
        fcvInterpolationType _fcvInterpolation, fcvBorderType _fcvBorder, int _fcvBorderValue) :
        cv::ParallelLoopBody(), src_data(_src_data), src_width(_src_width), src_height(_src_height), src_step(_src_step),
        dst_data(_dst_data), dst_width(_dst_width), dst_height(_dst_height), dst_step(_dst_step), type(_type),
        M(_M), fcvInterpolation(_fcvInterpolation),fcvBorder(_fcvBorder),
        fcvBorderValue(_fcvBorderValue) {}

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        uchar* dst = dst_data + range.start*dst_step;
        int rangeHeight = range.end - range.start;

        float rangeMatrix[9];
        rangeMatrix[0] = (float)(M[0]);
        rangeMatrix[1] = (float)(M[1]);
        rangeMatrix[2] = (float)(M[2]+range.start*M[1]);
        rangeMatrix[3] = (float)(M[3]);
        rangeMatrix[4] = (float)(M[4]);
        rangeMatrix[5] = (float)(M[5]+range.start*M[4]);
        rangeMatrix[6] = (float)(M[6]);
        rangeMatrix[7] = (float)(M[7]);
        rangeMatrix[8] = (float)(M[8]+range.start*M[7]);
        fcvWarpPerspectiveu8_v5(src_data, src_width, src_height, src_step, CV_MAT_CN(type), dst, dst_width, rangeHeight,
            dst_step, rangeMatrix, fcvInterpolation, fcvBorder, fcvBorderValue);
    }

    private:
    const uchar*            src_data;
    const int               src_width;
    const int               src_height;
    const size_t            src_step;
    uchar*                  dst_data;
    const int               dst_width;
    const int               dst_height;
    const size_t            dst_step;
    const int               type;
    const double*           M;
    fcvInterpolationType    fcvInterpolation;
    fcvBorderType           fcvBorder;
    int                     fcvBorderValue;

    FcvWarpPerspectiveLoop_Invoker(const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
    const FcvWarpPerspectiveLoop_Invoker& operator= (const FcvWarpPerspectiveLoop_Invoker &);  // = delete;
};

int fastcv_hal_warpPerspective(
    int             src_type,
    const uchar*    src_data,
    size_t          src_step,
    int             src_width,
    int             src_height,
    uchar*          dst_data,
    size_t          dst_step,
    int             dst_width,
    int             dst_height,
    const double    M[9],
    int             interpolation,
    int             border_type,
    const double    border_value[4])
{
    // Do not support inplace case
    if (src_data == dst_data)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Inplace is not supported");

    // The input channel should be 1
    if (CV_MAT_CN(src_type) != 1)
        CV_HAL_RETURN_NOT_IMPLEMENTED("Multi-channels is not supported");

    INITIALIZATION_CHECK;

    fcvStatus               status = FASTCV_SUCCESS;
    fcvBorderType           fcvBorder;
    uint8_t                 fcvBorderValue = 0;
    fcvInterpolationType    fcvInterpolation;

    switch (border_type)
    {
        case cv::BorderTypes::BORDER_CONSTANT:
        {
            if ((border_value[0] == border_value[1]) &&
                (border_value[0] == border_value[2]) &&
                (border_value[0] == border_value[3]))
            {
                fcvBorder       = fcvBorderType::FASTCV_BORDER_CONSTANT;
                fcvBorderValue  = static_cast<uint8_t>(border_value[0]);
                break;
            }
            else
                CV_HAL_RETURN_NOT_IMPLEMENTED("Different border value is not supported");
        }
        case cv::BorderTypes::BORDER_REPLICATE:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_REPLICATE;
            break;
        }
        case cv::BorderTypes::BORDER_TRANSPARENT:
        {
            fcvBorder = fcvBorderType::FASTCV_BORDER_UNDEFINED;
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Border type:%s is not supported", borderToString(border_type)));
    }

    switch(interpolation)
    {
        case cv::InterpolationFlags::INTER_NEAREST:
        {
            fcvInterpolation = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
            break;
        }
        default:
            CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Interpolation type:%s is not supported",
                                          interpolationToString(interpolation)));
    }

    int nThreads = cv::getNumThreads();
    int nStripes = nThreads > 1 ? 3*nThreads : 1;

    if(CV_MAT_DEPTH(src_type) == CV_8U)
    {
        cv::parallel_for_(cv::Range(0, dst_height),
            FcvWarpPerspectiveLoop_Invoker(src_data, src_width, src_height, src_step, dst_data, dst_width, dst_height,
            dst_step, src_type, M, fcvInterpolation, fcvBorder, fcvBorderValue), nStripes);
    }
    else
        CV_HAL_RETURN_NOT_IMPLEMENTED(cv::format("Src type:%s is not supported", cv::typeToString(src_type).c_str()));

    CV_HAL_RETURN(status, hal_warpPerspective);
}