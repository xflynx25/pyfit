# script that interactively should take you through this "semi" autoML process


"""
1. prompt for your folder 
2. read the database from there, start working on database preprocessing
    2a. if no database, prompt for which default database you'd like
    2b. would you like me to screw up the database a bit to practice cleaning and stuff
3. ask if you want specific eda tasks or just defaults, then let you know where you can view results
    3a. alos give feature engineering suggestions
4. allow you to do your transformer stuff, feature engineering 
    4a. optional cycle through step 3 repeatedly
5. offer up some models, customize it through questions, as well as how to feature select, how to hyperparam, and how to train (how long etc)
6. train model, save evaluation metrics & the model 
    6a. optional hop back to 3 or 5, train more models if want 
7. compare trained models on the test set if you'd like.

This whole process should be somewhat encapsulated in lab
"""

print("Semiauto is coming soon")