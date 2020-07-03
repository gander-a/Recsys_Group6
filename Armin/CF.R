library(clue)
library(MLmetrics)
library(scales)

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
sample <- sample.int(n = nrow(input), size = floor(.75*nrow(input)), replace = F)
training <- data[sample, ]
test  <- data[-sample, ]

para = c(10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
lossfunctions = c("l2")
summary = matrix(NA, 1, length(para))
for (i in 1:length(para)) {
  
#Clustering
buckets = para[i]
clust = kmeans(training, buckets)

t = 3
input_train = cbind(clust$cluster, data_ids$tweet_id[sample], timestamps[sample,c(t)])
colnames(input_train) = c("user", "item", "rating")
input_train = as.data.frame(input_train)

#Recosystem Training
train_data = data_memory(user_index = input_train$user, item_index = input_train$item, rating = input_train$rating)

r = Reco()
opts_tune = r$tune(train_data, opts = list(dim = c(10L, 20L),
                               costp_l1 = c(0, 0.1),
                               costp_l2 = c(0.01, 0.1),
                               costq_l1 = c(0, 0.1),
                               costq_l2 = c(0.01, 0.1),
                               lrate = c(0.01, 0.1))
)

r$train(train_data, opts = c(opts_tune$min, nthread = 1, niter = 20))

#Recosystem Testing
test_clustmem = as.numeric(cl_predict(clust, newdata = test))

input_test = cbind(test_clustmem, data_ids$tweet_id[-sample], timestamps[-sample,c(t)])
colnames(input_test) = c("user", "item", "rating")
input_test = as.data.frame(input_test)

test_data = data_memory(user_index = input_test$user, item_index = input_test$item, rating = input_test$rating)

pred = r$predict(test_data, out_memory())

summary[i]=PRAUC(pred, input_test$rating)
print(summary[i])
}

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
