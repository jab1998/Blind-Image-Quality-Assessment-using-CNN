function feat= dataSetFeat(file)
imDis=imread(file);
templateModel = load('templateModel.mat');
templateModel = templateModel.templateModel;
mu_prisparam = templateModel{1};
cov_prisparam = templateModel{2};
meanOfSampleData = templateModel{3};
principleVectors = templateModel{4};

feat = computefeatures(imDis,mu_prisparam,cov_prisparam,principleVectors,meanOfSampleData);
