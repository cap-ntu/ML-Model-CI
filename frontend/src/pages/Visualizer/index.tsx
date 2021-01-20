import React from 'react';
import { Graphviz  } from 'graphviz-react';
import { Row, Col, Card, Button, Select, Form } from "antd";
import axios from 'axios';
import { config } from 'ice';

const defaultOptions = {
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

const onFinish = (values: any) => {
	console.log('Success:', values);
};

const onFinishFailed = (errorInfo: any) => {
	console.log('Failed:', errorInfo);
};


export default class Visualizer extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			data: '',
			isFetching: false
		};
	}

	componentDidMount() {
		this.setState({ isFetching: true });
		const targetUrl = config.visualizerURL;
		axios.get(`${targetUrl}/${this.props.match.params.id}`)
			.then(res => {
				this.setState({ data: res.data.dot });
			})
		}

	render() {
		return (
			<Row gutter={16}>
				<Col span={16}>
					<Card title={`Model Structure`} bordered={false}>
						<div id="graphviz">
						<Graphviz dot={this.state.data ? this.state.data : `graph {}`} options={defaultOptions} />
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
