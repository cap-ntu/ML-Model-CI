import React from 'react';
import { ReactD3GraphViz } from '@hikeman/react-graphviz';
import { Row, Col, Card, Button, Select, Form } from 'antd';
import axios from 'axios';
import { config } from 'ice';
import { GraphvizOptions } from 'd3-graphviz';

const defaultOptions: GraphvizOptions = {
  fit: true,
  height: 1000,
  width: 1000,
  zoom: true,
  zoomScaleExtent: [0.5, 20],
  zoomTranslateExtent: [[-2000,-20000],[1000, 1000]]
};

const layout = {
  labelCol: { span: 8 },
  wrapperCol: { span: 16 },
};

const tailLayout = {
  wrapperCol: { span: 16, offset: 8 },
};

const onFinish = (values: unknown) => {
  console.log('Success:', values);
};

const onFinishFailed = (errorInfo: unknown) => {
  console.log('Failed:', errorInfo);
};

type VisualizerProps = { match: any};
type VisualizerState = { data: string; isLoaded: boolean };
export default class Visualizer extends React.Component<VisualizerProps, VisualizerState> {
  constructor(props) {
    super(props);
    this.state = {
	  data: '',
	  isLoaded: false
    };
    this.show_layer_info = this.show_layer_info.bind(this);
  }

  public async componentDidMount() {
    const targetUrl = config.visualizerURL;
    const res = await axios.get(`${targetUrl}/${this.props.match.params.id}`);
    this.setState({
		  isLoaded: true,
		  data: res.data.dot
    });
  }

  public show_layer_info = (title: string)=>{
    if(title.includes('.weight')){
      const layer: string = title.replace('.weight','');
	  console.log(layer)	  
    }
  }


  public render() {
    return (
      <Row gutter={16}>
        <Col span={16}>
          <Card title="Model Structure" bordered={false}>
            <div id="graphviz">
			  <ReactD3GraphViz 
			  dot={this.state.isLoaded ? this.state.data : 'graph {}'} 
			  options={defaultOptions} 
			  onClick={this.show_layer_info}
			  />
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="Finetune" bordered={false}>
            <Form
              {...layout}
              name="basic"
              initialValues={{ remember: true }}
              onFinish={onFinish}
              onFinishFailed={onFinishFailed}
            >
              <Form.Item
                name="dataset"
                label="Select Dataset"
                rules={[{ required: true, message: 'Please select one of the datasets' }]}
              >
                <Select>
                  <Select.Option value="cifar-10">cifar-10</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item {...tailLayout}>
                <Button type="primary" htmlType="submit">
									Submit
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </Col>
      </Row>
    );

  };
};