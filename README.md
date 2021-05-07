## Creating a web app from the Quora Questions Pairs Kaggle competition 

My goal in this repo is to create a web application based on a solution from the [Quora Questions Pairs](https://www.kaggle.com/c/quora-question-pairs/overview) competition.
This competition required from the participants to create machine learning models that were able to classify if two questions, regardless of been asked in different ways, possess the same meaning.

From the solutions I found, all of them relied on the so called 'magic features'. These are features that exploit data leakage and others factors and could not be implemented in a production environment real world scenario. Competitors are allowed to use these 'tricks' in order to increase the performance of their models, but they do not suit my needs. 

Lucky me I found [this](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur/) post on Linkedin from [Abhishek Thakur](https://www.linkedin.com/in/abhi1thakur/) where he made several experiments without using the 'magic features'. His goal was to achieve near state-of-the-art accuracy with a very deep neural network architecture that took 15 hours to train. I will start more humble and implement some of his simpler experiments combined with some of my own and build up the complexity as I go.
