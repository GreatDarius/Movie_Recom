#!/usr/bin/env python
# coding: utf-8

# In[5]:


def combinedRecommender(t):
    
    #First Recommender based on the rating of movies
    def movieRecomBasedOnRating():

        import pandas as pd
        import numpy as np
        import difflib
        import sklearn as sk

        #Importing dataset and initiating Pre process
        #In my project I use a connection to database to retrieve data

        movie_recom_rating = pd.read_csv('downloads/movies.csv')

        #Getting some informations about our data, You can explore your data more

        #movie_recom_rating.head()
        #movie_recom_rating.info()
        #movie_recom_rating.shape


        #Getting the average of the votes
        MeanVote = movie_recom_rating['vote_average'].mean()
        
        #Getting 90% quantile of the number of the votes
        MinVote =  movie_recom_rating['vote_count'].quantile(0.90)

        #Getting the list of movies with number of votes more thanour minimum nuber of votes
        FilterMovies = movie_recom_rating.copy().loc[movie_recom_rating['vote_count']>=MinVote]

        #Calculating the average score of votes based on the weight of the number of voters
        def weighted_rating (x, MinVote=MinVote, MeanVote=MeanVote):
            voters = x['vote_count']
            avg_vote = x['vote_average']
            return (voters/(voters+MinVote)*avg_vote)+(MinVote/(MinVote+voters)*MeanVote)
        
        #List the movies based on the new scores
        FilterMovies['score'] = FilterMovies.apply(weighted_rating, axis=1)

        #Order them ascending
        FilterMovies=FilterMovies.sort_values('score', ascending = False)
        
        #Set the nimber of float to 3 values
        pd.set_option('precision',3)
        
        #Printing result for users
        return(FilterMovies[['title','vote_count','vote_average','score']].head(50))

    #second recommender based on the content of movies that user is intrested in
    def movieRecomBasedOnContent():
        import pandas as pd
        import numpy as np
        import difflib
        import sklearn as sk
        import array as arr

        #Importing dataset and initiating Pre process
        #In my project I use a connection to database to retrieve data
        
        movie_recom = pd.read_csv('downloads/movies.csv')

        #Getting some informations about our data set
        
        #movie_recom.head()
        #movie_recom.info()
        #movie_recom.shape

        #Feature selecting. I recommend to choose all the features of your data which has string in it

        features_selected= ['genres','keywords','overview','tagline','title','cast','crew','director']

        #Cleaning Data. replacing empty values with null.

        for item in features_selected:
            movie_recom[item]= movie_recom[item].fillna('')
        
        #Adding all the features to gether since we are going to use TFIDF to vectorize our data
        
        accumulated_features = movie_recom['genres']+' '+movie_recom['keywords']+' '+movie_recom['overview']+' '+movie_recom['tagline']+' '+movie_recom['title']+' '+movie_recom['cast']+' '+movie_recom['crew']+' '+movie_recom['director']

        #Data transformating. using TFIDF from scikit-learn of python we can vectorize our features in order to
        #feed them into cosine similarity function

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer =  TfidfVectorizer()
        features_vector = vectorizer.fit_transform(accumulated_features)

        #using cosine similarity function to provide needed values to measure the similarity of the contents
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_score = cosine_similarity(features_vector)

        #Input Data from user. we need to get the current movie which user is watching now.
        #for simplicity we can get the name of movie in form of input

        movie_name = input('What Movie You are Looking for? ')

        #Finding the closest match movies based on the cosine score
        
        list_of_movies_title = movie_recom['title'].tolist()
        find_close_match= difflib.get_close_matches(movie_name,list_of_movies_title)
        closest_match= find_close_match[0]

        #Finding the index of the movie with title

        indexOfTheMovie= movie_recom[movie_recom.title == closest_match]['index'].values[0]

        #Listing Similar movies

        similarity_score_list= list(enumerate(similarity_score[indexOfTheMovie]))

        #Movies with higher cosin score

        sortedSimilarMovies=sorted(similarity_score_list, key = lambda x:x[1], reverse = True)

        #Show top n movies to user

        i = 1 
        resultt=[]
        for movie in sortedSimilarMovies:
            index = movie[0]
            title_from_index = movie_recom[movie_recom.index == index]['title'].values[0]
            if (i<51):
                aa=(title_from_index)
                resultt.append(aa)
                i+=1
        return(resultt)
    
    #In this part based on the users input and wheather he/she is loged in or not we can use either of recommendation techniques
    #or we can combin them both
    #in the condition part of the if here we can check if user is loged in or not. I used numbers for simplicity.
    
    x = t
    if x == 1:
        res1=movieRecomBasedOnContent() 
        print(res1)
    elif x ==2:
        res2=movieRecomBasedOnRating()
        print(res2)
    else:
        result1=movieRecomBasedOnContent()
        result2=movieRecomBasedOnRating()
        print(result1)
        print(result2)


# In[8]:


combinedRecommender(3)


# In[ ]:





# In[ ]:





# In[ ]:




