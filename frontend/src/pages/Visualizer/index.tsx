import React from 'react';
import { Graphviz } from 'graphviz-react';
import { Row, Col, Card, Button, Select, Form, InputNumber, Switch, Input, Radio, Modal } from "antd";
import axios from 'axios';
import { config } from 'ice';


const defaultOptions = {
	fit: true,
	height: 1000,
	width: 1000,
	zoom: true,
	zoomScaleExtent: [0.5, 20],
	zoomTranslateExtent: [[-2000, -20000], [1000, 1000]]
};

const tailLayout = {
	wrapperCol: { span: 16, offset: 8 },
};

const onFinishFailed = (errorInfo: any) => {
	console.log('Failed:', errorInfo);
};

const defaultTrainerConfig = {
	'dataset_name': 'cifar-10',
	'num_epochs': 15,
	'batch_size': 8,
	'optimizer': 'Adam',
	'num_workers': 1,
	'lr': 0.01,
	'scheduler': 'StepLR',
	'loss_fn': 'CrossEntropyLoss',
	'optimizerConfig': {
		'beta1': 0.9,
		'beta2': 0.99,
		'eps': 1e-08,
		'weight_decay': 0,
		'amsgrad': 'False'
	}
}

const optimizerConfig = {
	'Adam': [
		{
			name: 'beta1',
			label: 'beta1',
			type: 'number',
			default: 0.9,
			min: 0,
			max: 1,
			step: 0.01,
			col: '8'
		},
		{
			name: 'beta2',
			label: 'beta2',
			type: 'number',
			default: '0.999',
			min: 0,
			max: 1,
			step: 0.001,
			col: '8'
		},
		{
			name: 'eps',
			label: 'eps',
			type: 'number',
			default: 1e-08,
			step: 1e-08,
			col: '8'
		},
		{
			name: 'weight_decay',
			label: 'weight_decay',
			type: 'number',
			default: 0,
			col: '12'
		},
		{
			name: 'amsgrad',
			label: 'amsgrad',
			type: 'boolean',
			default: false,
			col: 12
		}
	]
};

function constructInput(item: any) {
	switch (item.type) {
		case 'number':
			return (<InputNumber min={item.min} max={item.max} step={item.step} />)
		case 'boolean':
			return (
				<Radio.Group>
					<Radio.Button value="True">True</Radio.Button>
					<Radio.Button value="False">False</Radio.Button>
				</Radio.Group>)
		default:
			return (<Input />)
	}
}


function construcForm(item: object) {
	return (
		<Col span={item.col}>
		<Form.Item
			name={['optimizerConfig', item.name]}
			label={item.label}
			key={item.name}
		>
			{constructInput(item)}
		</Form.Item>
		</Col>
	)
}

export default class Visualizer extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			data: null,
			isFetching: false,
			optimizer: 'Adam',
			scheduler: 'stepLR',
			hyperParameterTuning: true
		};
		this.handleOptimizerChange = this.handleOptimizerChange.bind(this);
		this.handleSchedulerChange = this.handleSchedulerChange.bind(this);
		this.onFinish = this.onFinish.bind(this)
	}

	async onFinish(values: object) {
		values['tuning'] = this.state.hyperParameterTuning
		// TODO(JSS) add model structure modifications to value
		// TODO(JSS) add job page to display training status
		let response = await axios.post(`${config.trainerURL}/${this.props.match.params.id}`, values)
		console.log('Success:', response.data.jobID);
		Modal.success({
			title: 'Finetune Job Created',
			content:
					<h6>Your finetune job is submitted successfully<br />
				You can visit the following link to check the training status<br />
				<a href={`/job/${response.data.jobID}`}>Job: {response.data.jobID}</a>
					</h6>
		});

	};

	handleOptimizerChange = (optimizer: string) => {
		this.setState({
			optimizer: optimizer
		})
	};

	handleSchedulerChange = (scheduler: string) => {
		this.setState({
			scheduler: scheduler
		})
	};

	handleHyperParameterSwitch = (checked: boolean) => {
		this.setState({
			hyperParameterTuning: checked
		})
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
							name="basic"
							initialValues={defaultTrainerConfig}
							onFinish={this.onFinish}
							onFinishFailed={onFinishFailed}
						>
							<Form.Item
								name="dataset_name"
								label="Name of Dataset"
								rules={[{ required: true, message: 'Please select one of the datasets' }]}
							>
								<Select>
									<Select.Option value="cifar-10">cifar-10</Select.Option>
								</Select>
							</Form.Item>
							<Form.Item
								name="num_epochs"
								label="Number of Epochs"
								rules={[{ required: true, message: 'Please set your number of epochs' }]}
							>
								<InputNumber min={1} />
							</Form.Item>
							<Form.Item
								name="batch_size"
								label="Batch Size"
								rules={[{ required: true, message: 'Please set your batch size' }]}
							>
								<InputNumber min={1} />
							</Form.Item>
							<Form.Item
								name="num_workers"
								label="Maxmium Number of CPU workers"
							>
								<InputNumber min={1} />
							</Form.Item>
							<Form.Item label="Automated HyperParameter Tuning">
								<Switch checked={this.state.hyperParameterTuning} onChange={this.handleHyperParameterSwitch} value={this.state.hyperParameterTuning} />
							</Form.Item>
							{
								this.state.hyperParameterTuning ? '' :

									[<Form.Item
										name="optimizer"
										label="Optimizer"
										key="optimizer"
									>
										<Select value={this.state.optimizer} onChange={this.handleOptimizerChange}>
											<Select.Option value="Adam">Adam</Select.Option>
										</Select>
									</Form.Item>
									]
										.concat(<Row>{optimizerConfig[this.state.optimizer].map(construcForm)}</Row>)
										.concat(
											[
												<Form.Item
													name="lr"
													label="Initial learning rate"
													key="lr"
												>
													<InputNumber min={0} />
												</Form.Item>,
												<Form.Item
													name="scheduler"
													label="Type of learning rate scheduler"
													key="scheduler"
												>
													<Select value={this.state.scheduler} onChange={this.handleSchedulerChange}>
														<Select.Option value="StepLR">StepLR</Select.Option>
													</Select>
												</Form.Item>,
												<Form.Item
													name="loss_fn"
													label="Type of loss function"
													key="loss_fn"
												>
													<Select>
														<Select.Option value="CrossEntropyLoss">Cross-Entropy Loss</Select.Option>
													</Select>
												</Form.Item>
											]
										)
							}
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
