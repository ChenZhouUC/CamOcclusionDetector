package main

import (
	"fmt"
	"occlusion/occlusion"

	"gocv.io/x/gocv"
)

func main() {

	/** ——————————————————————————
	* Occlusion Test
	* ——————————————————————————*/

	fmt.Println(" ——————> Occlusion Test <—————— ")
	// showWindow := gocv.NewWindow("Occlusion Test")
	imgPath := "/home/chenzhou/Documents/PythonRepo/AuxiliaryRepo/CamOcclusionDetect/data/Occluded/648.jpg"
	testImg := gocv.IMRead(imgPath, 1)
	occlScore := occlusion.ComplexityDetector(&testImg)
	fmt.Printf("==> TestImg: %s \n==> TestScore: %v \n", imgPath, occlScore)
	fmt.Println(" ——————> Test Closed <—————— ")

	// showWindow.IMShow(testImg)
	// showWindow.WaitKey(-1)

}
