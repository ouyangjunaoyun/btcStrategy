import pickle
import datetime as dt
import numpy as np
from irc import dio
import os
# Datefunctions

cwd = "/v/global/user/j/ju/junouyan/Jun/pylib/"

class DateFunctions:

    @staticmethod
    def prepareBusinessDate(startDate = dt.date(2003,1,1), endDate =  dt.date.today()):
        gw = dio.GW()
        return dio.get_business_days(gw.US, startDate,endDate)
        # pass

    # businessDayList = prepareBusinessDate()
    @staticmethod
    def isBusinessDay(date):
        return np.is_busday(date) & (date in businessDayList)

    @staticmethod
    def dumpBusinessDayToPickle(businessDayList, output_path = cwd + 'DateFunctions/businessDayList.pkl'):
        output = open(output_path, 'wb')
        pickle.dump(businessDayList, output)
        output.close()

    @staticmethod
    def loadBusinessDayFromPickle(output_path = cwd + 'DateFunctions/businessDayList.pkl'):
        pkl_file = open(output_path, 'rb')
        businessDayList = pickle.load(pkl_file)
        pkl_file.close()
        return businessDayList

    @staticmethod
    def getOffSetBUDay_(date, offset):
        new_date = date
        if offset == 0:
            return new_date if DateFunctions.isBusinessDay(date) else None
        elif offset > 0:
            while offset > 0:
                new_date += dt.timedelta(days =1)
                offset -= (1 if DateFunctions.isBusinessDay(new_date) else 0)
            return new_date
        else:
            while offset < 0:
                new_date -= dt.timedelta(days =1)
                offset += (1 if DateFunctions.isBusinessDay(new_date) else 0)
            return new_date
    @staticmethod
    def getOffSetBUDay(date, offset):
        if hasattr(date, '__len__'):
            return np.vectorize(DateFunctions.getOffSetBUDay_)(date, offset)
        else:
            return DateFunctions.getOffSetBUDay_(date, offset)

    @staticmethod
    def getAllBUDays():
        return businessDayList

# businessDayList = DateFunctions.prepareBusinessDate(startDate = dt.date(2003,1,1), endDate =  dt.date(2030,1,1))
# DateFunctions.dumpBusinessDayToPickle(businessDayList)
businessDayList = DateFunctions.loadBusinessDayFromPickle()