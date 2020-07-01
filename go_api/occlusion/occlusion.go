package occlusion

import (
	"image"
	"math"

	"gocv.io/x/gocv"
)

const (
	ResizeSetX     = 160
	ResizeSetY     = 120
	SuperpixelSetX = 12
	SuperpixelSetY = 9
	GaussianKernel = 5
	Thresh         = 50.0
)

func ComplexityDetector(src *gocv.Mat) float64 {
	// showWindow := gocv.NewWindow("Test")

	// resize to efficient size
	resized_mat := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV8UC3)
	resize_set := image.Point{X: ResizeSetX, Y: ResizeSetY}
	gocv.Resize(*src, &resized_mat, resize_set, 0, 0, 1)
	// extract value channel
	gocv.CvtColor(resized_mat, &resized_mat, gocv.ColorBGRToHSV)
	hsv_channel := gocv.Split(resized_mat)
	v_channel := hsv_channel[2]
	gaussian_kernel := image.Point{X: GaussianKernel, Y: GaussianKernel}
	gocv.GaussianBlur(v_channel, &v_channel, gaussian_kernel, 0, 0, gocv.BorderReplicate)
	// edge detection
	grad_x := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV16S)
	grad_y := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV16S)
	grad_x_abs := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV8U)
	grad_y_abs := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV8U)
	grad_abs := gocv.NewMatWithSize(ResizeSetY, ResizeSetX, gocv.MatTypeCV8U)

	gocv.Sobel(v_channel, &grad_x, 3, 1, 0, 3, 1, 0, 1)
	gocv.Sobel(v_channel, &grad_y, 3, 0, 1, 3, 1, 0, 1)
	gocv.ConvertScaleAbs(grad_x, &grad_x_abs, 1, 0)
	gocv.ConvertScaleAbs(grad_y, &grad_y_abs, 1, 0)
	gocv.AddWeighted(grad_x_abs, 0.5, grad_y_abs, 0.5, 0, &grad_abs)
	gocv.Threshold(grad_abs, &grad_abs, Thresh, 1, 0)
	// showWindow.IMShow(grad_abs)
	// showWindow.WaitKey(-1)

	w_num_pixel := float32(ResizeSetX) / float32(SuperpixelSetX)
	h_num_pixel := float32(ResizeSetY) / float32(SuperpixelSetY)
	score_mat := gocv.NewMatWithSize(SuperpixelSetY, SuperpixelSetX, gocv.MatTypeCV32F)

	for w := 0; w < SuperpixelSetX; w++ {
		for h := 0; h < SuperpixelSetY; h++ {
			crop := image.Rectangle{Min: image.Point{X: int(w_num_pixel * float32(w)), Y: int(h_num_pixel * float32(h))}, Max: image.Point{X: int(w_num_pixel * float32(w+1)), Y: int(h_num_pixel * float32(h+1))}}
			cropped := grad_abs.Region(crop)
			score_mat.SetFloatAt(h, w, float32(cropped.Mean().Val1))
		}
	}

	means := gocv.NewMatWithSize(1, 1, gocv.MatTypeCV64F)
	stds := gocv.NewMatWithSize(1, 1, gocv.MatTypeCV64F)
	gocv.MeanStdDev(score_mat, &means, &stds)
	divider := means.GetDoubleAt(0, 0) * 2
	var score float64
	if divider > 0 {
		score = stds.GetDoubleAt(0, 0) / divider
	} else {
		score = 1.0
	}

	return math.Min(score, 1.0)
}
