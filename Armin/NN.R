setwd("C:/Users/Armin/Desktop/Data Science/Recsys_Group6/Armin")
train <- read.csv("C:/Users/Armin/Desktop/Data Science/Recsys_Group6/Armin/dataset_users.csv")

#Preprocess
rem = c("X", "present_media", "present_domains", "text_tokens_tf", "text_tokens", "hashtags_tf", "hashtags", "tweet_type_StringIndex", "tweet_type_stringclassVec", "language_stringIndex", "language_stringclassVec", "features", "language_string")
tsnames = c("reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
ids = c("tweet_id", "enaging_user_id")
data = train[ , -which(names(train) %in% rem)]
timestamps = data[,tsnames]
data = data[,-c(2,3,4,5)]
data_ids = data[,which(names(data) %in% ids)]
data = data[,-which(names(data) %in% ids)]

#Create train and test set
set.seed(666)
sample <- sample.int(n = nrow(data), size = floor(.75*nrow(data)), replace = F)
training <- data[sample, ]
test  <- data[-sample, ]

para = c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
summary = matrix(NA, 1, length(para))
# for (i in 1:length(para)) {
  
#Clustering
buckets = 100#para[i]
clust = kmeans(training, buckets)

t = 4
traindata = cbind(training, clust$cluster, timestamps[sample, t])
#Training model
#kesas tensorflow model...
# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)

# Matrix
data = traindata

data <- as.matrix(data)
dimnames(data) <- NULL

# Partition
training <- data[,1:(ncol(data)-1)]
trainingtarget <- data[,(ncol(data))]

# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)

# Create Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 5, activation = 'relu', input_shape = (ncol(data)-1)) %>%
  layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

#Testing mddel
test_clustmem = as.numeric(cl_predict(clust, newdata = test))
testdata = cbind(test, test_clustmem, timestamps[-sample, t])

data = testdata

data <- as.matrix(data)
dimnames(data) <- NULL

# Partition
test <- data[,1:(ncol(data)-1)]
testtarget <- data[,(ncol(data))]

# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd)
test <- scale(test, center = m, scale = s)

# Evaluate
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)


#predict with keras tensorflow model...

summary[i]=PRAUC(pred, traindata$`timestamps[sample, t]`)
print(summary[i])

#}

#summary (like) 0.6467923 0.6562013 0.7150864 0.7440281 0.7501865 0.7512185 0.7697835 0.771053 0.76992 0.768791 
res_like = c(0.6578693, 0.7084555, 0.695253, 0.739171, 0.7526509, 0.7643586, 0.7618109, 0.7680769, 0.7686715, 0.7689876)
res_reply = c(0.0005486016, 0.0005486016, 0.0005506303, 0.02921141, 0.0005493117, 0.0005493112, 0.0005492805, 0.5008419, 0.5008418, 0.2565585)
res_retweet = c(0.1692298, 0.2763059, 0.166333, 0.26282, 0.3095574, 0.368526, 0.4091438, 0.5007259, 0.4982933, 0.5384007)
res_retweetcomment = c(0.00425845, 0.0001003033, 0.0001003033, 0.0001003033, 0.0001003033, 0.01478908, 0.0001002952, 0.00010028, 0.00010028, 0.05464512)

#Plot results
log10_minor_break = function (...){
  function(x) {
    minx         = floor(min(log10(x), na.rm=T))-1;
    maxx         = ceiling(max(log10(x), na.rm=T))+1;
    n_major      = maxx-minx+1;
    major_breaks = seq(minx, maxx, by=1)
    minor_breaks = 
      rep(log10(seq(1, 9, by=1)), times = n_major)+
      rep(major_breaks, each = 9)
    return(10^(minor_breaks))
  }
}

psize = 5
setoff = 0.04
tsize = 6
g = ggplot() +
  geom_line(aes(x=para, y = res_like, colour = "Like"), lwd = 1.5) +
  geom_line(aes(x=para, y = res_reply, colour = "Reply"), lwd = 1.5) +
  geom_line(aes(x=para, y = res_retweet, colour = "Retweet"), lwd = 1.5) +
  geom_line(aes(x=para, y = res_retweetcomment, colour = "Retweet+Comment"), lwd = 1.5) +
  geom_line(aes(x=para, y = 0.5), lwd = 1.5, col = "gray", linetype = "dashed") +
  
  geom_text(aes(x = para[which.max(res_like)], y = max(res_like)+setoff, label = as.character(round(max(res_like), digits = 4)), colour = "Like"), size = tsize) + 
  geom_point(aes(x = para[which.max(res_like)], y = max(res_like), colour = "Like"), size = psize) + 
  
  geom_text(aes(x = para[which.max(res_reply)], y = max(res_reply)+setoff, label = as.character(round(max(res_reply), digits = 4)), colour = "Reply"), size = tsize) + 
  geom_point(aes(x = para[which.max(res_reply)], y = max(res_reply), colour = "Reply"), size = psize) + 
  
  geom_text(aes(x = para[which.max(res_retweet)], y = max(res_retweet)+setoff, label = as.character(round(max(res_retweet), digits = 4)), colour = "Retweet"), size = tsize) + 
  geom_point(aes(x = para[which.max(res_retweet)], y = max(res_retweet), colour = "Retweet"), size = psize) + 
  
  geom_text(aes(x = para[which.max(res_retweetcomment)], y = max(res_retweetcomment)+setoff, label = as.character(round(max(res_retweetcomment), digits = 4)), colour = "Retweet+Comment"), size = tsize) + 
  geom_point(aes(x = para[which.max(res_retweetcomment)], y = max(res_retweetcomment), colour = "Retweet+Comment"), size = psize) + 
  
  scale_x_log10(minor_breaks=log10_minor_break())+
  annotation_logticks(side="b") + 
  theme_bw() +
  
  labs(y="PRAUC",
       x="Number of clusters (log scale)",
       title=sprintf("PRAUC vs. number of clusters")) +
  theme(panel.grid.minor = element_blank(),panel.background = element_blank(),
        axis.line = element_line(colour = "black"))+
  theme(text = element_text(size = 20)) +
  scale_colour_manual(name = "Class", breaks = c("Like", "Reply", "Retweet","Retweet+Comment"),
                      values = c("sienna1", "aquamarine2","slateblue2", "firebrick"))
plot(g)
pngname = sprintf("C:/Users/Armin/Desktop/Data Science/Recsys_Group6/Armin/PRAUC.png")
ggsave(pngname, width = 30, height = 20, units = "cm")
