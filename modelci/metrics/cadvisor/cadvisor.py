"""
Author: huangyz0918
Dec: A script to collect the information from cAdvisor monitors the Docker containers.
    Each request will return the last 1 minute statues of every running Docker container, 
    which has 60 status for each second. If we want to monitor the all information of 
    given containers we should using some database like Redis to store all the data and run
    this script every minutes.
Date: 17/04/2020
"""

import json
import requests

# Change default API informations here
DEFAULT_API_VERSION = 'v1.3'
DEFAULT_OBJECT = 'docker'
DEFAULT_LINK = 'http://localhost:8080/api/'


class CAdvisor(object):

    def __init__(self, base_link=None, version=None, query_object=None):
        """
        Parameters
        ----------
        base_link : str
            The basic API link for cAdvisor, example: 'http://localhost:8080/api/'.
        version : str
            The version of API, Supported API versions: v1.0,v1.1,v1.2,v1.3,v2.0,v2.1.
        query_object: str
            Request object for cAdvisor, can be 'docker' or 'machine'.
        """

        if base_link == None:
            self.base_link = DEFAULT_LINK
        else:
            self.base_link = base_link

        if version == None:
            self.version = DEFAULT_API_VERSION
        else:
            self.version = version

        if query_object == None:
            self.query_object = DEFAULT_OBJECT
        else:
            self.query_object = query_object

        self.api_link = self.base_link + self.version + '/'


    def machine_info(self):
        """
        Getting the machine information
        """

        return self.request_all('machine')


    def request_all(self, t_object=None):
        """
        Connect to cadvisor and get the last minute's worth of stats (should be 60 stats per container).

        Parameters
        ----------
        t_object : str
            Request object for cAdvisor, can be 'docker' or 'machine'.
        """

        if t_object == None:
            t_object = self.query_object
        elif t_object == 'machine':
            r = requests.get(self.api_link + t_object)
            return r.json()
        r = requests.get(self.api_link + t_object)
        raw_json = r.json()
        name = [raw_json[x]['name'] for x in raw_json]
        info = [raw_json[x] for x in raw_json]
        return dict(zip(name, info)) # return dict


    def request_without_cadvisor(self):
        """
        Return all the infomation from running containers expect cAdvisor's.
        """

        raw_json = self.request_all()
        name = [raw_json[x]['name'] for x in raw_json]
        info = [raw_json[x]
                       for x in raw_json if 'cadvisor' not in raw_json[x]['aliases']]
        return dict(zip(name, info)) # return dict


    def request_by_name(self, name):
        """
        Return all the infomation by given specific name, according to API, name can be id or image name.

        Parameters
        ----------
        name : str
            filter, name can be container id or image name.
        """

        raw_json = self.request_all()
        names = [raw_json[x]['name'] for x in raw_json]
        info = [raw_json[x]
                       for x in raw_json if name in raw_json[x]['aliases']]
        return dict(zip(names, info)) # return dict


    def request_by_id(self, id):
        """
        Return all the infomation by given specific container id.

        Parameters
        ----------
        id : str
            Docker container's id, should be unique.
        """

        raw_json = self.request_all()
        name = [raw_json[x]['name'] for x in raw_json]
        info = [raw_json[x]
                       for x in raw_json if id == raw_json[x]['id']]
        return dict(zip(name, info)) # return dict


    def request_by_image(self, image):
        """
        Return all the infomation by given specific container image name.

        Parameters
        ----------
        image: str
            Docker container's image name.
        """

        raw_json = self.request_all()
        name = [raw_json[x]['name'] for x in raw_json]
        info = [raw_json[x]
                       for x in raw_json if image == raw_json[x]['spec']['image']]
        return dict(zip(name, info)) # return dict


    def get_running_container(self, input_dict=None):
        """
        Return the basic infomation about running containers.

        Parameters
        ----------
        input_dict : dict
            extract information from the dict.
        """

        output_json = None
        if input_dict == None:
            input_dict = self.request_all()
        name = [input_dict[x]['name'] for x in input_dict]
        specs = [input_dict[x]['spec'] for x in input_dict]
        return dict(zip(name, specs))

    
    def get_stats(self, input_dict=None):
        """
        Return the stats infomation about running containers. Include metric like CPU, memory, etc.

        Parameters
        ----------
        input_dict : dict
            extract information from the dict.
        """

        output_json = None
        if input_dict == None:
            input_dict = self.request_all()
        name = [input_dict[x]['name'] for x in input_dict]
        specs = [input_dict[x]['stats'] for x in input_dict]
        return dict(zip(name, specs))     


    def get_specific_metrics(self, input_dict=None, metric=None):
        """
        Given a dict extract the information of specific metric

        Parameters
        ----------
        input_dict : dict
            extract information from the dict.
        metric: 
            metric field (diskio, cpu, diskio, memory, network, filesystem, task_stats, processes, accelerators).
        """

        if metric == None:
            metric = 'diskio'

        if input_dict == None:
            input_dict = self.request_all()

        io_dicts = []
        for container in input_dict.values():
            timestamps = [t['timestamp'] for t in container['stats']]
            diskios = [t[metric] for t in container['stats']]
            io_dict = dict(zip(timestamps, diskios))
            io_dicts.append(io_dict)
        return dict(zip(input_dict.keys(), io_dicts))


    def get_model_info(self, input_dict=None):
        """
        This function is getting the required data from cAdvisor for Hysia project.

        Parameters
        ----------
        input_dict : dict
            extract information from the dict.
        """
        if input_dict == None:
            input_dict = self.request_all()

        _names = [input_dict[x]['name'] for x in input_dict]
        _value_dict = [dict(id=input_dict[x]['id'],
                            creation_time=input_dict[x]['spec']['creation_time'],
                            image_name=input_dict[x]['spec']['image'],
                            has_cpu=input_dict[x]['spec']['has_cpu'],
                            cpu=input_dict[x]['spec']['cpu'],
                            has_memory=input_dict[x]['spec']['has_memory'],
                            memory=input_dict[x]['spec']['memory'],
                            stats=self._prepare_stat_dict(input_dict[x]['stats'])) for x in input_dict]
        _out_dict = dict(zip(_names, _value_dict))
        return _out_dict
    
    def _prepare_stat_dict(self, _stat_list):
        _stat_filter_list = []
        if 'accelerators' in _stat_list[0]:
            _stat_filter_list = [dict(timestamp=t['timestamp'],
                                    cpu=t['cpu'],
                                    memory=t['memory'],
                                    accelerators=t['accelerators']) for t in _stat_list]
        else:
            _stat_filter_list = [dict(timestamp=t['timestamp'],
                                    cpu=t['cpu'],
                                    memory=t['memory'],
                                    accelerators={}) for t in _stat_list]             
        return _stat_filter_list