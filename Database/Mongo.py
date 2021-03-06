import ssl
import time
import uuid
import typing
import requests
import pymongo

from decouple import config


class PhilterDB:

    def __init__(self):
        connection_string = config('mongo_connection')
        client = pymongo.MongoClient(connection_string,
                                     ssl_cert_reqs=ssl.CERT_NONE)

        self.__db = client["PHILTER_DB"]

    # Paginates the data to save on memory
    def __pagination(self,
                     limit: int,
                     table: str,
                     find_query: typing.Dict,
                     sort_query: typing.List,
                     page_num):

        skips = limit * page_num

        cursor = self.__db[table].find(find_query).sort(sort_query).skip(skips).limit(limit)

        return [x for x in cursor]

    # Logs in the user, if they are registered, otherwise registers them
    def login_user(self,
                   email: str,
                   name: str):

        data_filter = {"_id": email}
        updated_value = {"$set": {"name": name}}

        result = self.__db["USER_TABLE"].update_one(data_filter,
                                                    updated_value,
                                                    upsert=True)
        return result.matched_count

    # Saves a image to the database
    def save_image(self,
                   id: str,
                   title: str,
                   image_bytes: bytes,
                   **kwargs):

        url = config('save_image_url')
        api_key = config('api_key')

        payload = {"key": api_key,
                   "name": str(uuid.uuid4())}

        files = {"image": image_bytes}

        response = requests.post(url,
                                 data=payload,
                                 files=files)

        if response.json()['success']:

            image_dict = response.json()["data"]["image"]
            image_dict["_id"] = id
            image_dict.pop('name')
            image_dict["title"] = title
            image_dict["time"] = response.json()["data"]["time"]
            image_dict["delete_url"] = response.json()["data"]["delete_url"]

            for i in kwargs.keys():
                image_dict[i] = kwargs[i]

            doc = self.__db["IMAGE_TABLE"].insert_one(image_dict)

            return doc.inserted_id, response.json()["data"]["image"]["url"]
        else:
            return None

    # Saves a cluster of images to the database that belong to a specific user
    def save_cluster(self,
                     user_id: str,
                     image_list: typing.List,
                     algorithm: str,
                     tag: str):

        cluster_dict = {
            "userId": user_id,
            "imageList": image_list,
            "algorithm": algorithm,
            "tag": tag,
            "time": time.time() * 1000
        }

        return self.__db["CLUSTER_TABLE"].insert_one(cluster_dict).inserted_id

    # Gets a cluster for a user
    def get_clusters(self,
                     user_id,
                     limit,
                     sort_ascending: bool,
                     page_num=0,
                     **kwargs):

        query = {"userId": user_id}

        for i in kwargs.keys():
            query[i] = kwargs[i]

        print(query)

        sort_query = [("time", 1) if sort_ascending else ("time", -1)]
        print(sort_query)

        cluster_list = self.__pagination(limit,
                                         "CLUSTER_TABLE",
                                         query,
                                         sort_query,
                                         page_num)

        cluster_data = {"clusters": cluster_list,
                        "next_page": page_num + 1} if len(cluster_list) > 0 else {"clusters": cluster_list,
                                                                                  "next_page": None}

        return cluster_data

    # Converts a cluster full of image id's to image url's
    def cluster_to_image(self,
                         id_list: typing.List[str]):

        data = list(self.__db["IMAGE_TABLE"].find({"_id": {"$in": id_list}}))

        for idx, i in enumerate(data):
            if i["generated"]:
                break

        generated_list = [data.pop(idx)]

        return data, generated_list

    # Delete's a cluster for a user
    def delete_cluster(self,
                       table: str,
                       doc_id: str):

        value = self.__db[table].delete_one({"_id": doc_id})

        return value.deleted_count
