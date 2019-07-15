// Databricks notebook source
//CS6350 Assignment 1
//Nikhil Kalekar nlk180002
//Term: Summer 2019

//  Question 2 with multiple search terms


// COMMAND ----------

//  taking the textfiles in as RDDs 

val Plots = sc.textFile("/FileStore/tables/plot_summaries.txt")
val stopwordsRDD = sc.textFile(" /FileStore/tables/stopWords.txt")
val SearchTerms = sc.textFile("/FileStore/tables/UserSearch12.txt")


// COMMAND ----------

// each Id and its description is 1 entry in the new created plots RDD
Plots.take(1)

// COMMAND ----------

// These are  the stop words by which we will filter the movies RDD later.
stopwordsRDD.take(1)

// COMMAND ----------

//  this is the array of Search Terms or Search Query
SearchTerms.take(1)

// COMMAND ----------

// Using the RegEx for cleaning the movies RDD (i.e. to remove ;,'./;' etc)
val movieWords = Plots.map(row => row.replaceAll("""[\p{Punct}]""", " ").toLowerCase.split("""\s+"""))
movieWords.take(1)

// COMMAND ----------

// convert stopwords array to set so we can later filter movieWords RDD using the .contains function 

val stopWordsSet = stopwordsRDD.flatMap(s => s.split(",")).collect().toSet

stopwordsRDD.cache




// COMMAND ----------

// filtering the stop words
val cleanLine = movieWords.map(l => l.map(s => s).filter(s => stopWordsSet.contains(s) == false))

cleanLine.take(1)

cleanLine.cache


// COMMAND ----------

// convert our search terms into a list
// val trimmedList: List[String] = SearchTerms.collect().toList
val terms1 = SearchTerms.flatMap(_.split("\\s+"))
val terms = terms1.map(_.toLowerCase)
terms.take(3)

// COMMAND ----------

// val terms = sc.parallelize(trimmedList)

// terms.take(1)
//  taking the cartesian product of our search query with all the elements in our array
val cartesianProd = terms.cartesian(cleanLine)

cartesianProd.take(10)

// COMMAND ----------

// getting the key as search word and Doc Id
val SWDID = cartesianProd.map(a => ((a._1, a._2(0)), a._2))
SWDID.take(1)

// COMMAND ----------

// getting those documents where our search words have a Term Frequency and removing those where the term Frequency is 0
val termfreq = SWDID.map(l => (l._1, l._2.count(_.equals(l._1._1)).toDouble / l._2.size)).filter(c => c._2 != 0.0)

termfreq.take(2)


// COMMAND ----------

//  calculating the document frequency, i.e how many documents have the search term
val docFreq = termfreq.map(a => (a._1._1, 1)).reduceByKey(_ + _)

docFreq.collect()


// COMMAND ----------

val plotsCount = Plots.count()

// COMMAND ----------

// calculating the inverse Doc Frequency "idf"
val idfvalue = {
      docFreq.map(r => (r._1, Math.log(plotsCount / r._2)))
    }


idfvalue.collect()


// COMMAND ----------

// getting the search term as Key and value as (doc_id , TermFrequency)
val tfs = termfreq.map(a => ((a._1._1), (a._1._2, a._2)))

tfs.take(10)


// COMMAND ----------

// joining the above RDD to get the search term, Doc ID, Term Freq and the IDF in the same RDD so we can easily calculate the tf-idf for our search word in each document 

val tfidfjoin = tfs.join(idfvalue)

tfidfjoin.take(2)


// COMMAND ----------

// getting the search word and Doc_id, with the value as TF-IDF

val tfidf = tfidfjoin.map(a => ((a._1, a._2._1._1), (a._2._1._2 * a._2._2)))

tfidf.take(2)

// COMMAND ----------

//  getting the Movie_Meta.tsv data File to get the K,V pair as the DocID and the Movie Name :

val moviesFile = sc.textFile("/FileStore/tables/movie_metadata-ab497.tsv")

// map to get Movie Name as the Value and the Key as Document Id :
val movieMap = moviesFile.map(m=>(m.split("\t")(0), m.split("\t")(2)))

movieMap.take(2);

// COMMAND ----------

//  Getting the Size of the Search Terms. 
val Size = terms.count()

// COMMAND ----------

//  check if theres only 1 term eg. Comedy, Or is the search Term a long sentense  
if(Size==1){
//   getting the Key as Doc_ID and Value as TF-IDF
  val final1 = tfidf.map(a => (a._1._2, a._2))
  final1.take(2)
//    based on the Key, Join Movie MetaData and The Tf-IDF score
  val res = movieMap.join(final1)
  res.take(2)
//   Finally Map to a K,V such that K = Movie Name and Value is the Tf-IDF and sort according to the value of the K,V pair
  val finalres = res.map(a=>(a._2._1,a._2._2)).sortBy(-_._2).take(10)
  val last = sc.parallelize(finalres)
  last.take(10)
}else{
//   getting the frequency count of the search terms to calculate cosine similarity
  val countQuery = terms.map(x => (x, 1))
  countQuery.take(5)
  val tfsQuery = countQuery.reduceByKey(_ + _)
  val tfQuery = tfsQuery.map(x => (x._1, x._2.toDouble / Size))
  tfQuery.collect()
//   join frequency of word with the idf value: to get K,V as: K= Search term and V= tf-Idf, SearchTerm Term Freq  
  val tfidfset = idfvalue.join(tfQuery)
  tfidfset.collect()
//    now getting the K as SearchTerm , V as termFreq * tf-IDF 
  val tfidfQuery = tfidfset.map(a => (a._1, a._2._1 * a._2._2))
  tfidfQuery.collect()
//   grabbing the tfidf values K= Searchterm V= (DocId, tf-idf)
  val tfidf1 = tfidf.map(a => ((a._1._1), (a._1._2, a._2)))
  tfidf1.collect()
//   tfidfjoin will be K= SearchTerm , V= DocID, tf-idf, termFreq * tf-IDF
  val tfidfjoin = tfidf1.join(tfidfQuery)
//   getting K,V as K= DocID, V=(searchTerm, tf-idf)
  val tfidfprod = tfidfjoin.map(a => ((a._2._1._1), (a._1, a._2._1._2 * a._2._2)))
  tfidfprod.take(2)
//   getting K=DocID and V=tf-idf so that we can get Term Freq as same Docs, and then apply reduce to add up all the Term Frequency with same docs, i.e. if Action and Comedy have same Doc ID then by taking the Doc ID as KEy, We can Do reduceByKey , in this way we are combining various searchTerms based on same Document ID.
  val tfidfonlyprod = tfidfprod.map(a => (a._1, a._2._2))
  tfidfonlyprod.take(2)
  val final1 = tfidfonlyprod.reduceByKey(_ + _)
  final1.take(2)
//   taking the movie names
  val res = movieMap.join(final1)
  res.take(2)
//  getting the top 10 movies with the SearchTerms 
  val finalres = res.map(a=>(a._2._1,a._2._2)).sortBy(-_._2).take(10)
  val last = sc.parallelize(finalres)
  last.take(12)
  }

// COMMAND ----------





// COMMAND ----------


