//Open ROI path
//roi_path = File.openDialog("Choose Roi File");
//roiManager("Open", roi_path);
n = roiManager("count");

//Cycle through Rois and renames them. only works less than 1000 right now. 
for (index=0; index<n; index++) {
roinum_temp = index+1;
roinum = "" + roinum_temp;
//roinum = roinum_temp.substring(1,roinum_temp.length());
roiManager("Select", index);
roiManager("Rename",roinum);
}