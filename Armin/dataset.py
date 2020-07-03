from pyspark.sql.functions import array_contains, expr, lit, size, split, when, isnan, isnull
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, StructField, StructType, StringType
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Word2Vec, HashingTF, IDF


TRAIN_DATA_PATH = "hdfs:///user/pknees/RSC20/training.tsv"
VALIDATION_DATA_PATH = "hdfs:///user/pknees/RSC20/val.tsv"
COLNAMES = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",
			"tweet_type","language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",
			"engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",
			"enaging_user_id", "enaging_user_follower_count", "enaging_user_following_count",
			"enaging_user_is_verified", "enaging_user_account_creation", "engagee_follows_engager",
			"reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]
SCHEMA = StructType([
	StructField("text_tokens", ArrayType(StringType())),
	StructField("hashtags", ArrayType(StringType())),
	StructField("tweet_id", StringType()),
	StructField("present_media", ArrayType(StringType())),
	StructField("present_links", ArrayType(StringType())),
	StructField("present_domains", ArrayType(StringType())),
	StructField("tweet_type", StringType()),
	StructField("language", StringType()),
	StructField("tweet_timestamp", IntegerType()),
	StructField("engaged_with_user_id", StringType()),
	StructField("engaged_with_user_follower_count", IntegerType()),
	StructField("engaged_with_user_following_count", IntegerType()),
	StructField("engaged_with_user_is_verified", BooleanType()),
	StructField("engaged_with_user_account_creation", IntegerType()),
	StructField("enaging_user_id", StringType()),
	StructField("enaging_user_follower_count", IntegerType()),
	StructField("enaging_user_following_count", IntegerType()),
	StructField("enaging_user_is_verified", BooleanType()),
	StructField("enaging_user_account_creation", IntegerType()),
	StructField("engagee_follows_engager", BooleanType()),
	StructField("reply_timestamp", IntegerType()),
	StructField("retweet_timestamp", IntegerType()),
	StructField("retweet_with_comment_timestamp", IntegerType()),
	StructField("like_timestamp", IntegerType()),
])


class Dataset:

	def __init__(self, spark_session, path):
		self.spark_session = spark_session
		self.path = path
		self.dataframe = self.load_dataframe()

	def load_dataframe(self, sep="\x01"):
		print("Loading dataframe...")
		df = self.spark_session.read.csv(self.path, sep=sep)
		# Replace column names: the validation dataset does not have the last 4 columns as defined in COLNAMES
		df = df.toDF(*(COLNAMES if len(df.columns) == len(COLNAMES) else COLNAMES[:-4]))
		return df

	def shape(self):
		return self.dataframe.count(), len(self.dataframe.columns)

	def _split_column_values(self, column, sep="\t"):
		self.dataframe = self.dataframe.withColumn(column, split(column, sep))

	def _tokenize_hashtags(self, sep="\t", replace=True):
		hashtag_tokenizer = RegexTokenizer(inputCol="hashtags", outputCol="hashtags_list", pattern="\t")
		self.dataframe = hashtag_tokenizer.transform(self.dataframe.fillna("empty", subset=["hashtags"]))
		if replace:
			self.dataframe = self.dataframe.drop("hashtags").withColumnRenamed("hashtags_list", "hashtags")

	def split_list_columns(self, replace=True):
		"""Split values of list attributes parsed as string."""
		for col in [COLNAMES[i] for i in [0, 3, 4, 5]]:
			print("Converting string values in {} to arrays...".format(col))
			self._split_column_values(col)
		self._tokenize_hashtags(replace=replace)

	def cast_datatypes(self):
		for struct in SCHEMA if len(self.dataframe.columns) == len(SCHEMA) else SCHEMA[:-4]:
			self.dataframe = self.dataframe.withColumn(struct.name, self.dataframe[struct.name].cast(struct.dataType))

	def split_media_attribute(self):
		"Split present_media values into multiple columns"
		for media in ['Photo', 'Video', 'GIF']:
			print("Generating new column: media_{}...".format(media.lower()))
			self.dataframe = self.dataframe.withColumn("media_{}".format(media.lower()),
													   when(array_contains("present_media", media), expr(
														   'size(filter(present_media, x -> x in ("{}")))'.format(
															   media))).otherwise(0))

	def _onehot_encode_boolean(self, column_name, replace=True):
		"""Onehot encode a column with boolean values"""
		print("Applying onehot encoding to {}".format(column_name))
		self.dataframe = self.dataframe.withColumn("{}_encoded".format(column_name),
												   when(self.dataframe[column_name] == True, 1).otherwise(0))
		if replace:
			self.dataframe = self.dataframe.drop(column_name).withColumnRenamed("{}_encoded".format(column_name),
																				column_name)

	def onehot_encode_boolean_attributes(self, replace=True):
		for name in [COLNAMES[i] for i in [12, 17, 19]]:
			self._onehot_encode_boolean(name, replace=replace)

	def _convert_list_to_size(self, column_name, replace=True):
		print("Creating new column: {}_count containing size of lists stored in {}".format(column_name, column_name))
		self.dataframe = self.dataframe.withColumn("{}_count".format(column_name),
												   when(size(column_name) == -1, 0).otherwise(size(column_name)))
		if replace:
			self.dataframe = self.dataframe.drop(column_name).withColumnRenamed("{}_count".format(column_name),
																				column_name)

	def convert_present_links(self, replace=True):
		self._convert_list_to_size("present_links", replace=replace)

	def _binary_encode_column(self, column_name, replace=True):
		print("Applying binary encoding on {}...".format(column_name))
		self.dataframe = self.dataframe.withColumn("{}_binary".format(column_name),
												   when(isnull(self.dataframe[column_name]), 0).otherwise(1))
		if replace:
			self.dataframe = self.dataframe.drop(column_name).withColumnRenamed("{}_binary".format(column_name),
																				column_name)

	def binary_encode_target_attributes(self):
		for name in COLNAMES[-4:]:
			self._binary_encode_column(name)

	def label_encode_categorical(self, colnames, replace=True):
		for colname in colnames:
			print("Applying StringIndex on {}...".format(colname))
			string_indexer = StringIndexer(inputCol=colname, outputCol="{}_encoded".format(colname)).setHandleInvalid(
				"keep")
			model = string_indexer.fit(self.dataframe)
			self.dataframe = model.transform(self.dataframe)
			if replace:
				self.dataframe = self.dataframe.drop(colname).withColumnRenamed("{}_encoded".format(colname), colname)

	def tokenize_text_attributes(self, colname_num_features, method="TFIDF", replace=True):
		if method == "TFIDF":
			for cn in colname_num_features:
				colname, num_features = cn
				print("Generating term frequencies for {}...".format(colname), end=" ")
				tf = HashingTF(inputCol=colname, outputCol="{}_tf".format(colname), numFeatures=num_features)
				self.dataframe = tf.transform(self.dataframe)
				print("Done!\nGenerating idf values...", end=" ")
				idf = IDF(inputCol="{}_tf".format(colname), outputCol="{}_idf".format(colname))
				self.dataframe = idf.fit(self.dataframe).transform(self.dataframe)
				print("Done!")
				if replace:
					self.dataframe = self.dataframe.drop(colname).withColumnRenamed("{}_idf".format(colname), colname)

	def preprocess(self):
		self.split_list_columns()
		self.cast_datatypes()
		self.split_media_attribute()
		self.onehot_encode_boolean_attributes()
		self.convert_present_links()
		# Target attributes do not exist in validation dataframe
		if set(COLNAMES[-4:]).issubset(set(self.dataframe.columns)):
			self.binary_encode_target_attributes()
		self.tokenize_text_attributes([("text_tokens", 2^20), ("hashtags", 2^5)])
		self.label_encode_categorical(['tweet_type', 'language'])

	def _scale_column(self, column, replace=True):
		print("Scaling {}...".format(column))
		unlist = udf(lambda x: round(float(list(x)[0]), 3), DoubleType())
		assembler = VectorAssembler(inputCols=[column], outputCol=column + '_vectorized')
		scaler = MinMaxScaler(inputCol="{}_vectorized".format(column), outputCol="{}_scaled".format(column))
		pipeline = Pipeline(stages=[assembler, scaler])
		self.dataframe = pipeline.fit(self.dataframe).transform(self.dataframe).withColumn(column + "_scaled",
																							  unlist(column+ "_scaled")).drop(column + "_vectorized")
		if replace:
			self.dataframe = self.dataframe.drop(column).withColumnRenamed("{}_scaled".format(column), column)

	def scale(self, columns):
		for column in columns:
			self._scale_column(column)

