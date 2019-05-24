run("Close All");

dirsource = getDirectory("Choose Source Directory");

list= getFileList(dirsource);

n=lengthOf(list);
dir = dirsource;
dirsave = dir+"timecourses/";
File.makeDirectory(dirsave);
setBatchMode(false);

dirrois = getDirectory("Choose ROIs Directory");
listrois= getFileList(dirrois);
roiname = listrois[0];

for (index=0; index<n; index++) {
	if (endsWith(list[index],".tif")){

	fname = list[index];
	open(dir+fname);
	rename("Original");
	//numimgs=nSlices;
	
	//Extract ROI timecourses
	roiManager("Open", dirrois+roiname);
	roiManager("Show All");
	run("Set Measurements...", "  mean redirect=None decimal=3");
	roiManager("Multi Measure");
	saveAs("Results", dirsave+list[index]+"_tc.txt");
	run("Select All");
	roiManager("Deselect");
	roiManager("Delete");

	run("Close All");
	}
}
