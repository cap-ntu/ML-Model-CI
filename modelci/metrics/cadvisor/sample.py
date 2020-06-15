from modelci.metrics.cadvisor.cadvisor import CAdvisor

# Tests
if __name__ == "__main__":
    # Init the CAdvisor client, you can set the API information here.
    cAdvisor = CAdvisor()

    # base_link=None, version=None, query_object=None
    cAdvisor_version12 = CAdvisor(version='v1.2')  # default is v1.3
    raw_all_data_12 = cAdvisor_version12.request_all()
    # print("All raw data from v1.2 API: \n", raw_all_data_12)

    # Get the local machine information
    machine_info = cAdvisor.machine_info()
    # print('Machine info: \n', machine_info)

    # Request all the data from API, return a dict
    all_data = cAdvisor.request_all()
    all_stat = cAdvisor.get_stats()

    # You can request information by name or an id
    all_data_cadvisor = cAdvisor.request_by_name("cadvisor")
    all_stat_cadvisor = cAdvisor.get_stats(all_data_cadvisor)
    # print("All the stats from cadvisor: \n", all_data_cadvisor)

    # You can also request all the container information without cAdvisor
    data_all_without_cadvisor = cAdvisor.request_without_cadvisor()
    # print("All the container's data expect cAdvisor: \n", data_all_without_cadvisor)

    # Request some specific metrics by setting the field.
    # The field can be: diskio, cpu, diskio, memory, network, filesystem, task_stats, processes
    data_diskio_all = cAdvisor.get_specific_metrics(metric='diskio')
    # print('All diskio data: \n', data_diskio_all)

    # You can also using JSON filter if you only want some specific container information
    all_information_ubuntu = cAdvisor.request_by_image("ubuntu:16.04")
    data_cpu_ubuntu = cAdvisor.get_specific_metrics(input_dict=all_information_ubuntu, metric='cpu')
    # print('All CPU data related to containers from ubuntu 16.04 Docker image: \n', data_cpu_ubuntu)

    # You can also get the basic information from all the running containers
    data_running_containers = cAdvisor.get_running_container()
    # The function can also received a filtered JSON
    all_information_by_id = cAdvisor.request_by_id("afcb380e62bbec4fb7992cda3c986d387b5bc137fdea7dc0d1d13448012d5a5d")
    data_running_container_by_id = cAdvisor.get_running_container(all_information_by_id)
    # print('Basic info about container afcb380e62bbec4fb7992cda3c986d387b5bc137fdea7dc0d1d13448012d5a5d: \n', data_running_container_by_id)

    # You can use json to dump dict data to JSON easily
    # print(json.dumps(all_data))
    # with open('example_data.json', 'w') as f:
    #     json.dump(all_data, f)

    # Tests about getting required format data
    client = CAdvisor(base_link='')  # link to your remote
    tensorflow_resnet_latest = client.request_by_image('bitnami/tensorflow-serving:latest')
    out = client.get_model_info(tensorflow_resnet_latest)
    # with open('example_tf_resnet.json', 'w') as f:
    #     json.dump(out, f)
