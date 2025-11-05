import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MySQLdb
from sklearn.svm import SVR
import datetime
import time

# Connect to MySQL
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="prediction_v3")
cur = db.cursor()

def listToString(s): 
    return ''.join(s)

while True:
    df = pd.read_csv("niteditedfinal.csv")
    df = df.fillna(0)

    # Features and labels
    X = df.iloc[:, 0:5].values
    y_temp = df.iloc[:, 5].values
    y_ghi = df.iloc[:, 6].values

    # Train SVR models
    svr_temp = SVR(kernel='rbf')  
    svr_ghi = SVR(kernel='rbf')

    svr_temp.fit(X, y_temp)
    svr_ghi.fit(X, y_ghi)

    # Predict for current time + 15 minutes
    next_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
    time_str = next_time.strftime("%Y-%m-%d %H:%M")
    now = list(map(int, next_time.strftime("%Y,%m,%d,%H,%M").split(",")))

    # Predict temperature and GHI
    temp = svr_temp.predict([now])[0]
    ghi = svr_ghi.predict([now])[0]

    # Power calculation
    f = 0.18 * 7.4322 * ghi
    insi = temp - 25
    midd = 0.95 * insi
    power = f * midd

    print("Power:", power)
    print("Temperature:", temp)
    print("GHI:", ghi)

    # Insert into DB
    sql = """INSERT INTO svr_prediction (time_updated, Temperature, GHI, power) VALUES (%s, %s, %s, %s)"""
    try:
        print("Writing to the database...")
        cur.execute(sql, (time_str, float(temp), float(ghi), float(power)))
        db.commit()
        print("Write complete")
    except Exception as e:
        db.rollback()
        print("We have a problem:", e)

    
