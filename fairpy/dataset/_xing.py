# @Author  : Peizhao Li <peizhaoli05@gmail.com>
# @License : Apache License 2.0

""" Code adopted from https://github.com/MilkaLichtblau/xing_dataset/blob/master/src/XingProfilesReader.py """

import os
import sys
import uuid
import glob
import math
import json
import datetime
import pandas as pd
from typing import Sequence, Union

from ._base import TabDataset


class Candidate():
    """
    represents a candidate in a set that is passed to a search algorithm
    a candidate composes of a qualification and a list of protected attributes (strings)
    if the list of protected attributes is empty/null this is a candidate from a non-protected group
    natural ordering established by the qualification
    """

    def __init__(self, qualification, protectedAttributes):
        """
        @param qualification : describes how qualified the candidate is to match the search query
        @param protectedAttributes: list of strings that represent the protected attributes this
                                    candidate has (e.g. gender, race, etc)
                                    if the list is empty/null this is a candidate from a non-protected group
        """
        self.__qualification = qualification
        self.__protectedAttributes = protectedAttributes
        # keeps the candidate's initial qualification for evaluation purposes
        self.__originalQualification = qualification
        self.uuid = uuid.uuid4()

    @property
    def qualification(self):
        return self.__qualification

    @qualification.setter
    def qualification(self, value):
        self.__qualification = value

    @property
    def originalQualification(self):
        return self.__originalQualification

    @originalQualification.setter
    def originalQualification(self, value):
        self.__qualification = value

    @property
    def isProtected(self):
        '''
        true if the list of ProtectedAttribute elements actually contains anything
        false otherwise
        '''
        return not self.__protectedAttributes == []


class Xing(TabDataset):
    """ https://github.com/MilkaLichtblau/xing_dataset """

    file2url = {
        "SHAano01_1-40.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano01_1-40.json",
        "SHAano02-41-80.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano02-41-80.json",
        "SHAano03_81-120.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano03_81-120.json",
        "SHAano04_121-160.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano04_121-160.json",
        "SHAano05_161-200.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano05_161-200.json",
        "SHAano06_201-240.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano06_201-240.json",
        "SHAano07_241-280.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano07_241-280.json",
        "SHAano08_281-320.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano08_281-320.json",
        "SHAano09_321-360.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano09_321-360.json",
        "SHAano10_361-400.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano10_361-400.json",
        "SHAano11_401-440.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano11_401-440.json",
        "SHAano12_441-480.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano12_441-480.json",
        "SHAano13_481-520.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano13_481-520.json",
        "SHAano14_521-560.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano14_521-560.json",
        "SHAano15_561-600.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano15_561-600.json",
        "SHAano16_601-640.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano16_601-640.json",
        "SHAano17-641-680.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano17-641-680.json",
        "SHAano18_681-720.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano18_681-720.json",
        "SHAano19_721-760.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano19_721-760.json",
        "SHAano20_761-800.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano20_761-800.json",
        "SHAano21_801-840.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano21_801-840.json",
        "SHAano22_841-880.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano22_841-880.json",
        "SHAano23_881-920.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano23_881-920.json",
        "SHAano24_921-960.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano24_921-960.json",
        "SHAano25_961-1000.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano25_961-1000.json",
        "SHAano26_1001-1040.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano26_1001-1040.json",
        "SHAano27_1041-1080.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano27_1041-1080.json",
        "SHAano28_1081-1120.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano28_1081-1120.json",
        "SHAano29_1121-1160.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano29_1121-1160.json",
        "SHAano30_1161-1200.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano30_1161-1200.json",
        "SHAano31_1201-1240.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano31_1201-1240.json",
        "SHAano32_1241-1280.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano32_1241-1280.json",
        "SHAano33_1281-1320.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano33_1281-1320.json",
        "SHAano34_1320-1360.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano34_1320-1360.json",
        "SHAano35-1361-1400.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano35-1361-1400.json",
        "SHAano36-1401-1440.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano36-1401-1440.json",
        "SHAano37_1441-1480.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano37_1441-1480.json",
        "SHAano38_1481-1520.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano38_1481-1520.json",
        "SHAano39_1521-1560.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano39_1521-1560.json",
        "SHAano40_1561-1600.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano40_1561-1600.json",
        "SHAano41_1601-1640.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano41_1601-1640.json",
        "SHAano42_1641-1680.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano42_1641-1680.json",
        "SHAano43_1681-1720.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano43_1681-1720.json",
        "SHAano44_1721-1760.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano44_1721-1760.json",
        "SHAano45_1760-1800.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano45_1760-1800.json",
        "SHAano46_1801-1840.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano46_1801-1840.json",
        "SHAano47_1841-1880.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano47_1841-1880.json",
        "SHAano48_1881-1920.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano48_1881-1920.json",
        "SHAano49_1921-1960.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano49_1921-1960.json",
        "SHAano50_1961-2000.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano50_1961-2000.json",
        "SHAano51_2001-2040.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano51_2001-2040.json",
        "SHAano52_2041-2080.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano52_2041-2080.json",
        "SHAano53_2081-2120.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano53_2081-2120.json",
        "SHAano54_2121-2160.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano54_2121-2160.json",
        "SHAano55_2161-2200.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano55_2161-2200.json",
        "SHAano56_2201-2240.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano56_2201-2240.json",
        "SHAano57_2241-2280.json": "https://raw.githubusercontent.com/MilkaLichtblau/xing_dataset/master/data/SHAano57_2241-2280.json",
    }

    EDUCATION_OR_JOB_WITH_NO_DATES = 3  # months count if you had a job that has no associated dates
    EDUCATION_OR_JOB_WITH_SAME_YEAR = 6  # months count if you had a job that started and finished in the same year
    EDUCATION_OR_JOB_WITH_UNDEFINED_DATES = 1  # month given that the person entered the job

    def __init__(
            self,
            sen_feat: Union[str, Sequence[str]] = "sex",
            dir_path: str = os.path.join(sys.path[1], "fairpy/data/xing"),
            download: bool = False,
    ):
        self._check_and_download_data(dir_path, self.file2url, download)

        self.entireDataSet = pd.DataFrame(columns=['protected', 'nonProtected', 'originalOrdering'])

        files = glob.glob(dir_path + "/*.json")

        for filename in files:
            print(filename)
            key, protected, nonProtected, origOrder = self.__readFileOfQuery(filename)
            self.entireDataSet.loc[key] = [protected, nonProtected, origOrder]

        print(self.entireDataSet["originalOrdering"]["Dental Assistant"][20].qualification)

    def __readFileOfQuery(self, filename):
        """
        takes one .json file and reads all information, creates candidate objects from these
        information and sorts them into 3 arrays. One contains all protected candidates, one contains
        all non-protected candidates, one contains all candidates in the same order as they appear
        in the json-file
        @param filename: the json's filename
        @return:
            key: the search query string
            protected: array that contains all protected candidates
            nonProtected: array that contains all nonProtected candidates
        """

        protected = []
        nonProtected = []
        originalOrdering = []

        currentfile = open(filename)
        data = json.load(currentfile)

        xingSearchQuery = data['category']
        # if the Xing search query results in a gender neutral list,
        # we take female as the protected attribute
        protectedAttribute = 'm' if data['dominantSexXing'] == 'f' else 'f'

        for r in data['profiles']:
            # determine Member since / Hits
            if 'memberSince_Hits' in r['profile'][0]:
                hits_string = r['profile'][0]['memberSince_Hits']
                hits = hits_string.split(' / ')[1]
            else:
                hits = 1

            work_experience = self.__determineWorkMonths(r)
            edu_experience = self.__determineEduMonths(r)
            score = (work_experience + edu_experience) * int(hits)

            if self.__determineIfProtected(r, protectedAttribute):
                protected.append(Candidate(score, [protectedAttribute]))
                originalOrdering.append(Candidate(score, [protectedAttribute]))
            else:
                nonProtected.append(Candidate(score, []))
                originalOrdering.append(Candidate(score, []))

        protected.sort(key=lambda candidate: candidate.qualification, reverse=True)
        nonProtected.sort(key=lambda candidate: candidate.qualification, reverse=True)

        self.__normalizeQualifications(protected + nonProtected)
        self.__normalizeQualifications(originalOrdering)

        currentfile.close()

        return xingSearchQuery, protected, nonProtected, originalOrdering

    def __normalizeQualifications(self, ranking):
        # find highest qualification of candidate
        qualifications = [ranking[i].qualification for i in range(len(ranking))]
        highest = max(qualifications)
        for candidate in ranking:
            candidate.qualification = candidate.qualification / highest
            candidate.originalQualification = candidate.originalQualification / highest

    def __determineIfProtected(self, r, protAttr):
        """
        takes a JSON profile and finds if the person belongs to the protected group
        Parameter:
        ---------
        r : JSON node
        a person description in JSON, everything below node "profile"
        """

        if 'sex' in r['profile'][0]:
            if r['profile'][0]['sex'] == protAttr:
                return True
            else:
                return False
        else:
            print('>>> undetermined\n')
            return False

    def __determineWorkMonths(self, r):
        """
        takes a person's profile as JSON node and computes the total amount of work months this
        person has
        Parameters:
        ----------
        r : JSON node
        """

        total_working_months = 0  # ..of that profile
        job_duration = 0

        if len(r['profile'][0]) > 4:  # a job is on the profile
            list_of_Jobs = r['profile'][0]['jobs']
            # print('profile summary' + str(r['profile'][0]['jobs']))
            for count in range(0, len(list_of_Jobs)):
                if len(list_of_Jobs[count]) > 3:  # an exact duration is given at 5 nodes!

                    job_duration_string = list_of_Jobs[count]['jobDates']
                    if job_duration_string == 'bis heute':
                        # print('job with no dates found - will be count for ' + str(job_with_no_dates) + ' months.')
                        job_duration = self.EDUCATION_OR_JOB_WITH_NO_DATES

                    else:
                        job_start_string, job_end_string = job_duration_string.split(' - ')

                        if len(job_start_string) == 4:
                            job_start = datetime.datetime.strptime(job_start_string, "%Y")
                        elif len(job_start_string) == 7:
                            job_start = datetime.datetime.strptime(job_start_string, "%m/%Y")
                        else:
                            print("error reading start date")

                        if len(job_end_string) == 4:
                            job_end = datetime.datetime.strptime(job_end_string, "%Y")
                        elif len(job_end_string) == 7:
                            job_end = datetime.datetime.strptime(job_end_string, "%m/%Y")
                        else:
                            print("error reading end date")

                        if job_end - job_start == 0:
                            delta = self.EDUCATION_OR_JOB_WITH_SAME_YEAR
                        else:
                            delta = job_end - job_start

                        job_duration = math.ceil(delta.total_seconds() / 2629743.83)

                total_working_months += job_duration
        else:
            print('-no jobs on profile-')

        return total_working_months

    def __determineEduMonths(self, r):
        """
        takes a person's profile as JSON node and computes the total amount of work months this
        person has
        Parameters:
        ----------
        r : JSON node
        """

        total_education_months = 0  # ..of that profile
        edu_duration = 0

        if 'education' in r:  # education info is on the profile
            list_of_edu = r['education']  # edu child nodes {institution, url, degree, eduDuration}
            # print('education summary' + str(r['education']))
            for count in range(0, len(list_of_edu)):
                if 'eduDuration' in list_of_edu[count]:  # there are education dates

                    edu_duration_string = list_of_edu[count]['eduDuration']
                    if edu_duration_string == ('bis heute' or None or ''):
                        edu_duration = self.EDUCATION_OR_JOB_WITH_NO_DATES
                    else:
                        edu_start_string, edu_end_string = edu_duration_string.split(' - ')

                        if len(edu_start_string) == 4:
                            edu_start = datetime.datetime.strptime(edu_start_string, "%Y")
                        elif len(edu_start_string) == 7:
                            edu_start = datetime.datetime.strptime(edu_start_string, "%m/%Y")
                        else:
                            print("error reading start date")

                        if len(edu_end_string) == 4:
                            edu_end = datetime.datetime.strptime(edu_end_string, "%Y")
                        elif len(edu_end_string) == 7:
                            edu_end = datetime.datetime.strptime(edu_end_string, "%m/%Y")
                        else:
                            print("error reading end date")

                        if edu_end - edu_start == 0:
                            delta = self.EDUCATION_OR_JOB_WITH_SAME_YEAR
                        else:
                            delta = edu_end - edu_start

                        edu_duration = math.ceil(delta.total_seconds() / 2629743.83)

                        # print(job_duration_string)
                        # print('this job: ' + str(job_duration))

                else:
                    edu_duration = self.EDUCATION_OR_JOB_WITH_NO_DATES

                total_education_months += edu_duration
                # print('total jobs: ' + str(total_working_months))

            # print("studying: " + str(total_education_months))
        else:
            print('-no education on profile-')

        return total_education_months
