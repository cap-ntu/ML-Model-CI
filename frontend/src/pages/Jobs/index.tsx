import { Table, Card, Descriptions, Row, Col } from 'antd';
import React from 'react';
import reqwest from 'reqwest';
import { config } from 'ice';
import './index.css'

const columns = [
  {
    title: 'ID',
    dataIndex: '_id',
    sorter: true,
    width: '20%',
  },
  {
    title: 'Model',
    dataIndex: 'model',
    width: '20%',
  },
  {
    title: 'Status',
    dataIndex: 'status',
  },
];

const getRandomuserParams = params => ({
  results: params.pagination.pageSize,
  page: params.pagination.current,
  ...params,
});

export default class Jobs extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      data: [],
      pagination: {
        current: 1,
        pageSize: 10,
      },
      loading: false,
    };
    this.jobInfoExpand = this.jobInfoExpand.bind(this)
  }


  public componentDidMount() {
    const { pagination } = this.state;
    this.fetch({ pagination });
  }

  public handleTableChange = (pagination, filters, sorter) => {
    this.fetch({
      sortField: sorter.field,
      sortOrder: sorter.order,
      pagination,
      ...filters,
    });
  };

  public fetch = (params = {}) => {
    this.setState({ loading: true });
    reqwest({
      url: config.trainerURL,
      method: 'get',
      type: 'json',
      data: getRandomuserParams(params),
    }).then(data => {
      this.setState({
        loading: false,
        data,
        pagination: {
          ...params.pagination,
          total: data.length
        },
      });
    });
  };

  public parseProperty = (Property: any) =>{

    const properties = Object.keys(Property).filter((name) => (Property[name]!==null))
    return (
      properties.map(
        (item) => 
          <Row>
            <Col span={8}>{item}</Col>
            <Col span={6} offset={6}>{Property[item]}</Col>
          </Row>
      )
    )
  }

  public jobInfoExpand = (record)=>(
    <Descriptions 
      title="Job Details" 
      bordered
      contentStyle={{backgroundColor: '#fff'}}
    >
      <Descriptions.Item label="Dataset Name">
        {record.data_module.dataset_name}
      </Descriptions.Item>
      <Descriptions.Item label="Batch Size">
        {record.data_module.batch_size}
      </Descriptions.Item>
      <Descriptions.Item label="Num Workers">
        {record.data_module.num_workers}
      </Descriptions.Item>
      <Descriptions.Item label="Max Epochs">
        {record.max_epochs}
      </Descriptions.Item>
      <Descriptions.Item label="Min Epochs" span={2}>
        {record.min_epochs}
      </Descriptions.Item>
      <Descriptions.Item label="Loss Function" span={3}>
        {record.loss_function}
      </Descriptions.Item>
      <Descriptions.Item label="Learning Rate Scheduler">
        {record.lr_scheduler_type}
      </Descriptions.Item>
      <Descriptions.Item label="Learning Rate Scheduler Property" span={2}>
        {this.parseProperty(record.lr_scheduler_property)}
      </Descriptions.Item>
      <Descriptions.Item label="Optimizer">
        {record.optimizer_type}
      </Descriptions.Item>
      <Descriptions.Item label="Optimizer Property" span={2}>
        {this.parseProperty(record.optimizer_property)}
      </Descriptions.Item>
    </Descriptions>
  )


  public render() {
    const { data, pagination, loading } = this.state;
    return (
      <Card>
        <Table
          columns={columns}
          rowKey={record => record._id}
          dataSource={data}
          pagination={pagination}
          loading={loading}
          onChange={this.handleTableChange}
          expandable={{
            expandedRowRender: this.jobInfoExpand,
          }}
        />
      </Card>
    );
  }
}
