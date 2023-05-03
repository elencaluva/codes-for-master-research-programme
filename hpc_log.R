setwd("/home/julius_bs/xchen/")
#install.packages("doParallel",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/",lib="/home/julius_bs/xchen/R-4.2.2/library")
#install.packages("foreach",repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN/",lib="/home/julius_bs/xchen/R-4.2.2/library")
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
test<-datagenerator(100000,coeff,-1.08) 

#define gradient
grad<-c(50,100,300,500,750,1000,5000,10000,50000)


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



#logistic regression
s=Sys.time()
data_log<-data.frame(o3,s3,brier,sd3,o4,s4,sz,sd4,o11,s11,int,sd11,o12,s12,slope,sd12)
rep<-1000
train_log<-array(0,c(length(grad),rep,4))
test_log<-array(0,c(length(grad),rep,4))

for (i in 1:length(grad)){
  
  #for (j in 1:rep/mul){
  loglist<-foreach(k = 1:rep) %dopar%{
    trainset<-datagenerator(grad[i],coeff,-1.08)[,c(1:23,26)]
    log.null<-glm(event~1,data=trainset,family="binomial")
    log.full<-glm(event~.,data=trainset,family="binomial")
    log<-step(log.null,scope=list(lower=log.null, upper=log.full), trace=0, direction="both")
    
    #applied in itself
    trainset.log.predict<-predict(log,trainset,type = "response")
    #mul*(j-1)+k    
    tmp1<-c(mean((trainset[,24]-trainset.log.predict)^2),
            sum((trainset[,24]-trainset.log.predict)*(1-2*trainset.log.predict))/sqrt(sum(((1-2*trainset.log.predict)^2)*trainset.log.predict*(1-trainset.log.predict))),
            cox(trainset[,24],trainset.log.predict)[[2]],
            cox(trainset[,24],trainset.log.predict)[[1]])
    
    #applied in testset   
    test.log.predict<-predict(log,test,type = "response")
    tmp2<-c(mean((test[,26]-test.log.predict)^2),
            sum((test[,26]-test.log.predict)*(1-2*test.log.predict))/sqrt(sum(((1-2*test.log.predict)^2)*test.log.predict*(1-test.log.predict))),
            cox(test[,26],test.log.predict)[[2]],
            cox(test[,26],test.log.predict)[[1]])
    
    return(list(tmp1,tmp2))
  }#k=rep
  for(t in 1:rep){
    train_log[i,t,]<- loglist[[t]][[1]]
    test_log[i,t,]<- loglist[[t]][[2]]
  }
  #  }#j=rep/mul
  rm(loglist)
  gc()
  
  data_log[i,1]<- mean(train_log[i,,1]-test_log[i,,1])
  data_log[i,2]<- sd(train_log[i,,1]-test_log[i,,1])
  data_log[i,3]<- mean(test_log[i,,1])
  data_log[i,4]<- sd(test_log[i,,1])
  
  data_log[i,5]<- mean(train_log[i,,2]-test_log[i,,2])
  data_log[i,6]<- sd(train_log[i,,2]-test_log[i,,2])
  data_log[i,7]<- mean(test_log[i,,2])
  data_log[i,8]<- sd(test_log[i,,2])
  
  data_log[i,9]<- mean(train_log[i,,3]-test_log[i,,3])
  data_log[i,10]<- sd(train_log[i,,3]-test_log[i,,3])
  data_log[i,11]<- mean(test_log[i,,3])
  data_log[i,12]<- sd(test_log[i,,3])
  
  data_log[i,13]<- mean(train_log[i,,4]-test_log[i,,4])
  data_log[i,14]<- sd(train_log[i,,4]-test_log[i,,4])
  data_log[i,15]<- mean(test_log[i,,4])
  data_log[i,16]<- sd(test_log[i,,4])
  
}
e=Sys.time()
print(e-s)
test_log
train_log
data_log
write.table(data_log,"log.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(train_log,"train_log.csv",row.names=FALSE,col.names=TRUE,sep=",")
write.table(test_log,"test_log.csv",row.names=FALSE,col.names=TRUE,sep=",")