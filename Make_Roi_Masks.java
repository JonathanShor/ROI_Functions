//Open ROI path
roi_path = File.openDialog("Choose Roi File");
roiManager("Open", roi_path);
n = roiManager("count");

//Choose Avg Image
im_path = File.openDialog("Choose Avg Image");
open(im_path);

dirsavemaster = getDirectory("Choose a Directory to save your binary masks");

//Cycle through Rois and save masks as separate files 
for (index=0; index<n; index++) {
roiManager("Select", index);
run("Create Mask");
mask_num = 1000+index+1;
roiname = dirsavemaster+File.separator+"Mask_"+mask_num+".bmp";
saveAs("BMP", roiname);
close();
}