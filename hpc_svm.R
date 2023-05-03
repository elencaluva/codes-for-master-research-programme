setwd("/home/julius_bs/xchen/")
#install.packages("kernlab",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/",lib="/home/julius_bs/xchen/R-4.2.2/library")
library(caret,lib.loc="/home/julius_bs/xchen/R-4.2.2/library")
library(kernlab,lib.loc="/home/julius_bs/xchen/R-4.2.2/library")
library(foreach,lib.loc="/home/julius_bs/xchen/R-4.2.2/library")
library(doParallel,lib.loc="/home/julius_bs/xchen/R-4.2.2/library")

nslots <- Sys.getenv( "SLURM_CPUS_ON_NODE" )
print( nslots )
registerDoParallel(cores=as.numeric(nslots)/2)

#prespecify logistic regression coefficients
set.seed(12060929)
coeff<-rnorm(23,0,0.5)#0.7
#generate dataset 
datagenerator<-function(N,coeff,intercept){ #N=100, 300, 500, 1000, 3000, 5000
  #set.seed(12060929)
  X1<-rbinom(N,1,0.03) #prepare for calculation of intercept
  X2<-rbinom(N,1,0.06)
  X3<-rbinom(N,1,0.09)
  X4<-rbinom(N,1,0.12)
  X5<-rbinom(N,1,0.15)
  X6<-rbinom(N,1,0.18)
  X7<-rbinom(N,1,0.21)
  X8<-rbinom(N,1,0.24)
  X9<-rbinom(N,1,0.27)
  X10<-rbinom(N,1,0.3)
  X11<-rbinom(N,1,0.33)
  X12<-rbinom(N,1,0.36)
  X13<-rbinom(N,1,0.39)
  X14<-rbinom(N,1,0.42)
  X15<-rbinom(N,1,0.45)
  X16<-rbinom(N,1,0.48)
  X17<-rbinom(N,1,0.51)
  X18<-rbinom(N,1,0.54)
  X19<-rbinom(N,1,0.57)
  X20<-rbinom(N,1,0.6)
  X21<-rnorm(N,0.5,0.1)
  X22<-rnorm(N,0.5,0.5)
  X23<-rnorm(N,0.5,1)
  
  dat<-data.frame(X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,X19,X20,X21,X22,X23)
  dat$LP<-as.vector(as.matrix(dat)%*%as.matrix(coeff)+intercept)
  dat$y<-1/(1+exp(-dat$LP))
  dat$event<-as.numeric(runif(N)<dat$y)
  dat
}

#testset
set.seed(12060929)
test<-datagenerator(100000,coeff,-1.71) 

#define gradient
grad<-c(50,100,300,500,750,1000)

#50,100,
#brier score
o3<-rep(0,length(grad))
s3<-rep(0,length(grad))
brier<-rep(0,length(grad))
sd3<-rep(0,length(grad))
#Spiegelhalter z test
o4<-rep(0,length(grad))
s4<-rep(0,length(grad))
sz<-rep(0,length(grad))
sd4<-rep(0,length(grad))
#Cox’s intercept
o11<-rep(0,length(grad))
s11<-rep(0,length(grad))
int<-rep(0,length(grad))
sd11<-rep(0,length(grad))
#Cox’s slope
o12<-rep(0,length(grad))
s12<-rep(0,length(grad))
slope<-rep(0,length(grad))
sd12<-rep(0,length(grad))
#accuracy
accuracy<-rep(0,length(grad))


#cox slope & intercept----------------------------------------------------------
logit <-function (p){log(p/(1 - p))} 

cox = function(y, prob){
  dat <- data.frame(e = prob, o = y)
  dat$e[dat$e == 0] = 0.0000000001
  dat$e[dat$e == 1] = 0.9999999999
  dat$logite <- logit(dat$e)
  
  mfit = glm(formula = o~I(logite), 
             family = binomial(link = "logit"), dat)
  # browser()
  slope = mfit$coefficients[2]
  intercept = mfit$coefficients[1]
  return(list(slope = slope, intercept = intercept))
}

#spiegelhalter z ---------------------------------------------------------------
sh<-function(y,prob){
  prob[prob == 0] = 0.0000000001
  prob[prob == 1] = 0.9999999999
  s<-sum((y-prob)*(1-2*prob))/sqrt(sum((1-2*prob^2)*prob*(1-prob)))
  return (s)
}



#svm
s=Sys.time()
data_svm<-data.frame(o3,s3,brier,sd3,o4,s4,sz,sd4,o11,s11,int,sd11,o12,s12,slope,sd12)
rep<-50
train_svm<-array(0,c(length(grad),rep,4))
test_svm<-array(0,c(length(grad),rep,4))

for (i in 1:length(grad)){
  
  #for (j in 1:rep/mul){
  svmlist<-foreach(k = 1:rep) %dopar%{
    trainset<-datagenerator(grad[i],coeff,-1.71)[,c(1:23,26)]
    ctrl <- trainControl(method = "LOOCV", savePred=T,classProbs=TRUE)
    svm <- train(x=trainset[,1:23],
                     y=factor(trainset[,24],labels=c("noncase","case")),
                     method="svmLinear",
                     preProcess="scale",
                     trControl = ctrl, tuneGrid = data.frame(C=c(0.1,0.2,0.3,0.4,0.5,0.6,0.8,1) ) )
    #svm$pred[svm$pred[,"C"]==as.numeric(svm$bestTune),"noncase"]
    #applied in itself
    trainset.svm.predict<-as.numeric(predict(svm,trainset[,1:23]))-1
    #mul*(j-1)+k    
    tmp1<-c(mean((trainset[,24]-trainset.svm.predict)^2),
            sh(trainset[,24],trainset.svm.predict),
            cox(trainset[,24],trainset.svm.predict)[[2]],
            cox(trainset[,24],trainset.svm.predict)[[1]])

    #applied in testset   
    test.svm.predict<-as.numeric(predict(svm,test[,1:23]))-1
    tmp2<-c(mean((test[,26]-test.svm.predict)^2),
            sh(test[,26],test.svm.predict),
            cox(test[,26],test.svm.predict)[[2]],
            cox(test[,26],test.svm.predict)[[1]])

    return(list(tmp1,tmp2))
  }#k=rep
  for(t in 1:rep){
    train_svm[i,t,]<- svmlist[[t]][[1]]
    test_svm[i,t,]<- svmlist[[t]][[2]]
  }
  #  }#j=rep/mul
  #rm(svmlist)
  #gc()
  
  data_svm[i,1]<- mean(train_svm[i,,1]-test_svm[i,,1])
  data_svm[i,2]<- sd(train_svm[i,,1]-test_svm[i,,1])
  data_svm[i,3]<- mean(test_svm[i,,1])
  data_svm[i,4]<- sd(test_svm[i,,1])
  
  data_svm[i,5]<- mean(train_svm[i,,2]-test_svm[i,,2])
  data_svm[i,6]<- sd(train_svm[i,,2]-test_svm[i,,2])
  data_svm[i,7]<- mean(test_svm[i,,2])
  data_svm[i,8]<- sd(test_svm[i,,2])
  
  
  data_svm[i,9]<- mean(train_svm[i,,3]-test_svm[i,,3])
  data_svm[i,10]<- sd(train_svm[i,,3]-test_svm[i,,3])
  data_svm[i,11]<- mean(test_svm[i,,3])
  data_svm[i,12]<- sd(test_svm[i,,3])
  
  data_svm[i,13]<- mean(train_svm[i,,4]-test_svm[i,,4])
  data_svm[i,14]<- sd(train_svm[i,,4]-test_svm[i,,4])
  data_svm[i,15]<- mean(test_svm[i,,4])
  data_svm[i,16]<- sd(test_svm[i,,4])
}
e=Sys.time()
print(e-s)
data_svm
train_svm
test_svm
write.table(train_svm,"train_svm.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(test_svm,"test_svm.csv",row.names=FALSE,col.names=TRUE,sep=",")
