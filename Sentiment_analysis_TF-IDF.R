library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)
library(quanteda)

#Name files and call them in via file path 
file_a <- "~/Downloads/texts/A07594__Circle_of_Commerce.txt"
file_b <- "~/Downloads/texts/B14801__Free_Trade.txt"

#Read raw text files into R
text_a <- read_file(file_a)
text_b <- read_file(file_b)

#Combine texts into a tibble 
texts <- tibble(doc_title = c("Text A", "Text B"), 
                text = c(text_a, text_b))
texts

#Create texts copy to preserve original 
texts <- texts %>%
  mutate(text_original = text)

#Normalize the long s (ſ) to modern s
texts <- texts %>%   
  mutate(     
    text_clean = str_replace_all(text_original, "ſ", "s")   
  )  

#Check substitution 
tibble(
  long_s_before = str_count(texts$text_original, "ſ"),
  long_s_after  = str_count(texts$text_clean, "ſ")
)

##Tokenization 

#Tidytext built-in stopword list
data("stop_words")

#Add project-specific stopword(s)
custom_stopwords <- tibble(
  word = c("vnto","haue","doo","hath","bee","ye","thee","hee",
           "shall","hast","doe",
           "beene","thereof","thus")
)

all_stopwords <- bind_rows(stop_words, custom_stopwords) %>%
  distinct(word)

all_stopwords %>% slice(1:10)

#Tokenize texts, remove both standard and custom stopwords 
tokens <- texts %>%
  unnest_tokens(word, text_clean) %>%  
  filter(!str_detect(word, "[0-9]")) %>%
  filter(!str_detect(word, "[[:punct:][:symbol:]]"))  %>%
  mutate(word = str_to_lower(word)) %>%
  anti_join(all_stopwords, by = "word") 

#Label words as "positive" or "negative" 
bing <- get_sentiments("bing")

tokens_bing <- tokens %>% 
  inner_join(bing, by = "word")

tokens_bing

#Provide summary table with one row per document 
summary_tokens <- tokens_bing %>%
  group_by(doc_title) %>%
  summarise(
    positive_words = sum(sentiment == "positive"),
    negative_words = sum(sentiment == "negative"),
    net_sentiment = positive_words - negative_words
  )

summary_tokens

##Sentiment analysis with TF-IDF–weighted words

#Read raw text files into R
txt_circle  <- read_file("~/Downloads/texts/A07594__Circle_of_Commerce.txt")
txt_free <- read_file("~/Downloads/texts/B14801__Free_Trade.txt")
txt_third <- read_file("~/Downloads/texts/A06785.txt")

#Normalize the long s (ſ) to modern s
txt_circle <- str_replace_all(txt_circle, "ſ", "s")   
txt_free <- str_replace_all(txt_free, "ſ", "s")   
txt_third <- str_replace_all(txt_third, "ſ", "s") 

#Combine the texts  
texts <- c(
  "Circle_of_Commerce" = txt_circle,
  "Free_Trade"         = txt_free,
  "Third_Text_A06785"  = txt_third
)

corp <- corpus(texts)

##Tokenize and clean texts 

#Same pre-processing decisions as raw word count sentiment analysis 
toks <- tokens(
  corp,
  remove_punct   = TRUE,
  remove_numbers = TRUE,
  remove_symbols = TRUE
)

#Lowercase 
toks <- tokens_tolower(toks)

#Remove same standard and custom stopwords 
toks <- tokens_remove(
  toks, pattern = all_stopwords$word
  )

#Document-feature matrix (DFM)
dfm_mat <- dfm(toks)
dfm_mat

#Calculate TF-IDF 
dfm_tfidf <- dfm_tfidf(dfm_mat)

dfm_tfidf

##Convert the TF-IDF results to a tidy table 

#Convert TF-IDF to a data frame 
tfidf_df <- convert(
  dfm_tfidf,
  to = "data.frame"
)

#Convert wide TF-IDF data frame into tidy format 
tfidf_tidy <- tfidf_df %>%
  pivot_longer(
    cols = -doc_id, 
    names_to = "word", 
    values_to = "tfidf"
  )

tfidf_tidy

#Remove rows where TF-IDF weight is 0 
tfidf_tidy <- tfidf_tidy %>%
  filter(tfidf > 0)

#Keep Text A and Text B
tfidf_tidy <- tfidf_tidy %>%
  filter(doc_id %in% c("Circle_of_Commerce", "Free_Trade"))

tfidf_tidy

#Join the table to the sentiment dictionary 
#Keep sentiment-bearing words only 
tfidf_bing <- tfidf_tidy %>%
  inner_join(bing, by = "word")

tfidf_bing

#Highest weighted sentiment words
highest_sentiment_words <- tfidf_bing %>%
  group_by(doc_id) %>%
  slice_max(order_by = tfidf, n = 10) %>%
  arrange(doc_id, desc(tfidf))

highest_sentiment_words

#Weighted totals calculation 
tfidf_summary <- tfidf_bing %>%
  group_by(doc_id) %>%
  summarise(
    positive_tfidf = sum(tfidf[sentiment == "positive"]), 
    negative_tfidf = sum(tfidf[sentiment == "negative"]), 
    net_tfidf_sentiment = positive_tfidf - negative_tfidf 
  )

tfidf_summary

##Compare the raw counts sentiment analysis to the TF-IDF sentiment analysis 
#Rename raw-count result document names 
raw_summary <- summary_tokens %>%
  mutate(
    doc_id = case_when(
      doc_title == "Text A" ~ "Circle_of_Commerce",
      doc_title == "Text B" ~ "Free_Trade"
    )
  ) %>%
  select(
    doc_id,
    positive_words,
    negative_words,
    net_sentiment
  )

raw_summary

#Combine raw counts with TF-IDF Results 
final_comparison <- raw_summary %>% 
  left_join(tfidf_summary, by = "doc_id")

final_comparison