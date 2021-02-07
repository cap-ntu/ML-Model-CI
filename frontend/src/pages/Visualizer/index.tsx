/* eslint-disable no-underscore-dangle */
import React from 'react';
import { ReactD3GraphViz } from '@hikeman/react-graphviz';
import { Row, Col, Card, Popover, Button, Divider, Progress, Modal } from 'antd';
import axios from 'axios';
import { config } from 'ice';
import { GraphvizOptions } from 'd3-graphviz';
import GenerateSchema from 'generate-schema';
import Form from '@rjsf/material-ui';
import { ModelStructure, FinetuneConfig,DEFAULT_FINETUNE_CONFIG } from './utils/type'

// eslint-disable-next-line @typescript-eslint/no-var-requires
const mockStructure = require('./utils/mock.json');

const defaultOptions: GraphvizOptions = {
  fit: true,
  height: 700,
  width: 1000,
  zoom: true,
  zoomScaleExtent: [0.5, 20],
  zoomTranslateExtent: [[-2000, -20000], [1000, 1000]]
};

type VisualizerProps = { match: any };
type VisualizerState = {
  X: number;
  Y: number;
  currentX: number;
  currentY: number;
  graphData: string;
  isLoaded: boolean;
  isValidating: boolean;
  modelStructure: ModelStructure;
  finetuneConfig: FinetuneConfig;
  visible: boolean;
  currentLayerName: string;
  currentLayerInfo: object;
  layerSchema: object;
  configSchema: object;
  seconds: number;
};

export default class Visualizer extends React.Component<VisualizerProps, VisualizerState> {
  constructor(props) {
    super(props);
    this.state = {
      X: 0,
      Y: 0,
      currentX: 0,
      currentY: 0,
      seconds: 60,
      graphData: '',
      isLoaded: false,
      isValidating: false,
      modelStructure: { layer: {}, connection: {}},
      finetuneConfig: DEFAULT_FINETUNE_CONFIG,
      visible: false,
      currentLayerName: '',
      currentLayerInfo: {},
      layerSchema: {},
      configSchema: {
        'title': 'Finetune Settings',
        'type': 'object',
        'properties': {
          'dataset_name': {
            'type': 'string',
            default: 'CIFAR10',
            enum: ['CIFAR10']
          }
        }
      }
    };
    this.timer = 0;
    this.showLayerInfo = this.showLayerInfo.bind(this);
    this.layerSubmit = this.layerSubmit.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onLayerChange = this.onLayerChange.bind(this);
    this.onConfigChange = this.onConfigChange.bind(this);
    this.onClickValidate = this.onClickValidate.bind(this);
    this.countDown = this.countDown.bind(this);
  }

  public async componentDidMount() {
    const id: string = this.props.match.params.id;
    const graph = await axios.get(`${config.visualizerURL}/${id}`);
    this.setState({
      isLoaded: true,
      graphData: graph.data.dot,
      modelStructure: mockStructure
    });
  }

  /**
   * update layer info
   * @param layer event
   */
  public onLayerChange(layer: any){
    const properties = {...this.state.layerSchema.properties}
    Object.keys(properties).forEach(
      (property) => {
        properties[property].default = layer.formData[property]
      }
    )
  }

  /**
   * update finetine job config
   * TODO add more config options
   * @param e event
   */
  public onConfigChange = (config: any)=>{
	  this.setState(
      prevState =>
      {
        const newConfig = prevState.finetuneConfig
        // eslint-disable-next-line @typescript-eslint/camelcase
        newConfig.data_module = { ...config.formData, ...newConfig.data_module};
        return {finetuneConfig:prevState.finetuneConfig}
      }
    )
  }

  /**
   * record the mouse position
   * @param e event
   */
  public onMouseMove(e: any) {
    this.setState({ currentX: e.screenX, currentY: e.screenY });
  }

  /**
   * submit finetune job and modified model structures
   */
  public configSubmit = async () => {
    // submit model structure
    const layers = this.state.modelStructure.layer
    const updatedLayers = Object.keys(layers).reduce(function (r, e) {
      if (layers[e].op_ !== 'E') {
        r[e] = layers[e]
      }
      return r;
    }, {})
    // TODO add connection update info 
    const submittedStructure: ModelStructure = { 'layer': updatedLayers, 'connection': {} }
    let res = await axios.patch(`${config.structureRefractorURL}/${this.props.match.params.id}`, submittedStructure)
    // TODO submit training job
    const newConfig = {...this.state.finetuneConfig}
    newConfig.model = res.data.id;
    res = await axios.post(config.trainerURL, newConfig)
    Modal.success({
      title: 'Finetune Job Created',
      content:
					<h6>Your finetune job is submitted successfully<br />
				You can visit the following link to check the training status<br />
					  <a href='/jobs'>Job: {res.data.id}</a>
					</h6>
    });
  }

  /**
   * update model layer information after submit
   */
  public layerSubmit = (layer: any) => {
    const modifyMark = { op_: 'M' }
    const newConfig = { ...this.state.currentLayerInfo, ...layer.formData, ...modifyMark }
    const newStructure = { ...this.state.modelStructure }
    newStructure.layer[this.state.currentLayerName] = newConfig
    // close the form
    this.setState({ visible: false });
  }

  /**
   * display layer settings form after click on the graph
   * TODO add default value
   * 
   * @param title layer title
   * 
   */
  public showLayerInfo = (title: string) => {
    // TODO validation check of model layer name
    if (title.includes('.weight') || title.includes('.bias')) {
      const layerName: string = title.replace('.weight', '').replace('.bias', '');
      const layersInfo: object = this.state.modelStructure.layer;
      if (layerName in layersInfo && layerName !== this.state.currentLayerName) {
        const layerInfo = { ...layersInfo[layerName] }
        this.setState({ currentLayerInfo: layersInfo[layerName] })
        this.setState({ currentLayerName: layerName })
        delete layerInfo.op_;
        delete layerInfo.type_;
        const schema = GenerateSchema.json(`Layer ${layerName}`, layerInfo)
        delete schema.$schema
        Object.keys(schema.properties).forEach(
          (property) =>{
            schema.properties[property].default = layerInfo[property]
          }
        )
        // eslint-disable-next-line react/no-access-state-in-setstate
        this.setState({ X: this.state.currentX, Y: this.state.currentY })
        this.setState({ layerSchema: schema, visible: true, currentLayerName: layerName })
      }
    }
  }

  /**
   * display data during validating process
   */
  public onClickValidate = () =>{
    // TODO: pass parameters to validator
    this.setState({isValidating: true})
    this.setState({seconds: 60})
    this.timer = setInterval(this.countDown, 1000);
  }

  // refer to https://stackoverflow.com/a/40887181
  public countDown() {
    this.setState(
      prevState =>({ seconds: prevState.seconds - 1 })
    )
    if (this.state.seconds === 0) { 
      clearInterval(this.timer);
      this.setState({isValidating: false})
      // TODO: get validate accuracy
    }
  }

  public render() {
    return (
      <div onMouseMove={this.onMouseMove}>
        <Row gutter={16}>
          <Col span={15}>
            <Card bordered={false}>
              <h5 className="MuiTypography-root MuiTypography-h5">Model Structure</h5>
              <Divider />
              <div id="graphviz">
                <Popover
                  content={
                    <div>
                      <Form
                        schema={this.state.layerSchema}
                        onSubmit={this.layerSubmit}
                        onChange={this.onLayerChange}
                      />
                    </div>
                  }
                  title="Modify Layer Paramaters"
                  visible={this.state.visible}
                  placement="rightTop"
                  autoAdjustOverflow
                  overlayStyle={{left: this.state.X + 200, top: this.state.Y - 400}}
                  overlayInnerStyle={{width: 350, height: 600, overflowY: 'scroll'}}
                />
                <ReactD3GraphViz
                  dot={this.state.isLoaded ? this.state.graphData : 'graph {}'}
                  options={defaultOptions}
                  onClick={this.showLayerInfo}
                />
              </div>
            </Card>
          </Col>
          <Col span={7} offset={1}>
            <Card bordered={false}>
              <Form
                schema={this.state.configSchema}
                onSubmit={this.configSubmit}
                onChange={this.onConfigChange}
              >
                <div>
                  { this.state.isValidating || this.state.seconds===0?
                    <Card bordered={false}>
                      <Row justify="center">
                        <Progress
                          percent={100 - this.state.seconds/60*100}
                          format={()=>`${this.state.seconds} s`}
                        />
                      </Row>
                      <Row justify="center">
                        <h3>
                          Validation Accuracy :&nbsp;
                          <span className="ant-statistic-content">
                            {this.state.seconds===0?
                              `${0.00} %` // TODO replace with ajax data
                              : '  In Progress  '
                            }
                          </span>
                        </h3>
                      </Row>
                    </Card> : ''
                  }
                  <Row>
                    <Col span={11}>
                      <Button 
                        type="primary" 
                        htmlType="submit" 
                        size="large"
                        block
                      >
                        Train
                      </Button>
                    </Col>
                    <Col span={11} offset={2}>
                      
                      <Button 
                        type="primary" 
                        size="large" 
                        onClick={this.onClickValidate} 
                        disabled={this.state.isValidating}
                        block>
                        Validate
                      </Button>
                    </Col>
                  </Row>
                </div>
              </Form>
            </Card>
          </Col>
        </Row>
      </div>
    );

  };
};