/* eslint-disable no-underscore-dangle */
import React from 'react';
import {
  DagreReact,
  Node,
  Rect,
  ValueCache,
  ReportSize,
  Size,
  NodeOptions,
} from 'dagre-reactjs';
import {
  Row,
  Col,
  Card,
  Button,
  Divider,
  Progress,
  Modal,
  Tooltip,
  Dropdown,
  Space,
} from 'antd';
import axios from 'axios';
import { config } from 'ice';
import GenerateSchema from 'generate-schema';
import { withTheme } from '@rjsf/core';
import { Theme as AntDTheme } from '@rjsf/antd';
import 'antd/dist/antd.css';
import {
  SchemaForm,
  FormButtonGroup,
  FormEffectHooks,
  Submit,
} from '@formily/antd';
import { merge } from 'rxjs';
import { Input, Select, Upload, Switch } from '@formily/antd-components';
import { UncontrolledReactSVGPanZoom } from 'react-svg-pan-zoom';
import { CustomNodeLabel } from './components/CustomNode';
import {
  ModelStructure,
  FinetuneConfig,
  DEFAULT_FINETUNE_CONFIG,
  DEFAULT_CONFIG_SCHEMA,
} from './utils/type';

// eslint-disable-next-line @typescript-eslint/no-var-requires

const Form = withTheme(AntDTheme);
const components = {
  Input,
  Select,
  Upload,
  Switch,
};

const { onFieldValueChange$, onFieldInit$ } = FormEffectHooks;

type VisualizerProps = { match: any };
type VisualizerState = {
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
  graph: object;
};

export default class Visualizer extends React.Component<
VisualizerProps,
VisualizerState
> {
  constructor(props) {
    super(props);
    this.state = {
      seconds: 60,
      isLoaded: false,
      isValidating: false,
      modelStructure: { layer: {}, connection: {} },
      finetuneConfig: DEFAULT_FINETUNE_CONFIG,
      visible: false,
      currentLayerName: '',
      currentLayerInfo: {},
      layerSchema: {},
      configSchema: DEFAULT_CONFIG_SCHEMA,
      graph: {},
    };
    this.timer = 0;
    this.showLayerInfo = this.showLayerInfo.bind(this);
    this.layerSubmit = this.layerSubmit.bind(this);
    this.onLayerChange = this.onLayerChange.bind(this);
    this.onConfigChange = this.onConfigChange.bind(this);
    this.onClickValidate = this.onClickValidate.bind(this);
    this.countDown = this.countDown.bind(this);
  }

  public componentDidMount() {
    this.loadData();
  }

  /**
   * update layer info
   * @param layer event
   */
  public onLayerChange(layer: any) {
    const properties = { ...this.state.layerSchema.properties };
    Object.keys(properties).forEach((property) => {
      properties[property].default = layer.formData[property];
    });
  }

  /**
   * update finetine job config
   * TODO add more config options
   * @param e event
   */
  public onConfigChange = (config: any) => {
    this.setState((prevState) => {
      const newConfig = prevState.finetuneConfig;
      // eslint-disable-next-line @typescript-eslint/camelcase
      newConfig.data_module = { ...config.formData, ...newConfig.data_module };
      return { finetuneConfig: prevState.finetuneConfig };
    });
  };

  /**
   * load graph data
   */
  public loadData = () => {
    const id: string = this.props.match.params.id;
    axios.get(`${config.visualizerURL}/${id}`).then((res) => {
      const links = res.data.links.map((edge) => {
        edge.from = edge.source;
        edge.to = edge.target;
        delete edge.source;
        delete edge.target;
        return edge;
      });
      this.setState({
        isLoaded: true,
        graph: { nodes: res.data.nodes, links },
      });
    });

    axios.get(`${config.structureURL}/${id}`).then((res) => {
      this.setState({
        modelStructure: res.data,
      });
    });
  };

  /**
   * submit finetune job and modified model structures
   */
  public configSubmit = async () => {
    // submit model structure
    const layers = this.state.modelStructure.layer;
    const updatedLayers = Object.keys(layers).reduce(function (r, e) {
      if (layers[e].op_ !== 'E') {
        r[e] = layers[e];
      }
      return r;
    }, {});
    // TODO add connection update info
    const submittedStructure: ModelStructure = {
      layer: updatedLayers,
      connection: {},
    };
    let res = await axios.patch(
      `${config.structureRefractorURL}/${this.props.match.params.id}`,
      submittedStructure
    );
    // TODO submit training job
    const newConfig = { ...this.state.finetuneConfig };
    newConfig.model = res.data.id;
    res = await axios.post(config.trainerURL, newConfig);
    Modal.success({
      title: 'Finetune Job Created',
      content: (
        <h6>
          Your finetune job is submitted successfully
          <br />
          You can visit the following link to check the training status
          <br />
          <a href="/jobs">Job: {res.data.id}</a>
        </h6>
      ),
    });
  };

  /**
   * update model layer information after submit
   */
  public layerSubmit = (layer: any) => {
    const modifyMark = { op_: 'M' };
    const newConfig = {
      ...this.state.currentLayerInfo,
      ...layer.formData,
      ...modifyMark,
    };
    const newStructure = { ...this.state.modelStructure };
    newStructure.layer[this.state.currentLayerName] = newConfig;
    // close the form
    this.setState({ visible: false });
  };

  /**
   * display layer settings form after click on the graph
   * TODO add default value
   *
   * @param title layer title
   *
   */
  public showLayerInfo = (layerName: string) => {
    // TODO validation check of model layer name
    const layersInfo: object = this.state.modelStructure.layer;
    if (layerName in layersInfo && layerName !== this.state.currentLayerName) {
      const layerInfo = { ...layersInfo[layerName] };
      this.setState({ currentLayerInfo: layersInfo[layerName] });
      this.setState({ currentLayerName: layerName });
      delete layerInfo.op_;
      delete layerInfo.type_;
      const schema = GenerateSchema.json(`Layer ${layerName}`, layerInfo);
      delete schema.$schema;
      Object.keys(schema.properties).forEach((property) => {
        schema.properties[property].default = layerInfo[property];
      });
      // eslint-disable-next-line react/no-access-state-in-setstate
      this.setState({
        layerSchema: schema,
        visible: true,
        currentLayerName: layerName,
      });
    }
  };

  /**
   * display data during validating process
   */
  public onClickValidate = () => {
    // TODO: pass parameters to validator
    this.setState({ isValidating: true });
    this.setState({ seconds: 60 });
    this.timer = setInterval(this.countDown, 1000);
  };

  /**
   * refer to https://stackoverflow.com/a/40887181
   */
  public countDown() {
    this.setState((prevState) => ({ seconds: prevState.seconds - 1 }));
    if (this.state.seconds === 0) {
      clearInterval(this.timer);
      this.setState({ isValidating: false });
      // TODO: get validate accuracy
    }
  }

  /**
   * rewrite node render function
   * @param node
   * @param reportSize
   * @param valueCache
   * @param layoutStage
   */
  public renderNode = (
    node: NodeOptions,
    reportSize: ReportSize,
    valueCache: ValueCache,
    layoutStage: number
  ) => {
    return (
      <Node
        key={node.id}
        node={node}
        reportSize={reportSize}
        valueCache={valueCache}
        layoutStage={layoutStage}
        html
      >
        {{
          shape: (innerSize: Size) => (
            <Rect node={node} innerSize={innerSize} />
          ),
          label: () => (
            <CustomNodeLabel
              label={node.label}
              shape={node.meta.shape}
              onClick={() => this.showLayerInfo(node.meta.name)}
            />
          ),
        }}
      </Node>
    );
  };

  public render() {
    return (
      <div>
        <Row gutter={16}>
          <Col span={15}>
            <Card bordered={false}>
              <h5 className="MuiTypography-root MuiTypography-h5">
                Model Structure
              </h5>
              <Divider />
              <Dropdown
                visible={this.state.visible}
                overlayStyle={{
                  width: 300,
                  height: 400,
                  padding: 20,
                  overflowY: 'scroll',
                  backgroundColor: 'whitesmoke',
                }}
                overlay={
                  <Form
                    schema={this.state.layerSchema}
                    onSubmit={this.layerSubmit}
                    onChange={this.onLayerChange}
                  >
                    <Space>
                      <Button type="primary" htmlType="submit">
                      Submit
                      </Button>
                      <Button
                        htmlType="button"
                        onClick={()=>this.setState({ visible: false })}
                      >
                      Close
                      </Button>
                    </Space>

                  </Form>
                }
                trigger={['contextMenu', 'click']}
              >
                <div
                  id="graphviz"
                  style={{ position: 'relative', overflow: 'hidden' }}
                >
                  <div style={{ height: '100%' }}>
                    <UncontrolledReactSVGPanZoom
                      width={1000}
                      height={800}
                      background="#fff"
                    >
                      <svg width={500} height={15000}>
                        {this.state.isLoaded ? (
                          <DagreReact
                            nodes={this.state.graph.nodes}
                            edges={this.state.graph.links}
                            renderNode={this.renderNode}
                            defaultNodeConfig={{
                              styles: {
                                shape: {
                                  styles: { fill: '#845', strokeWidth: '2' },
                                },
                              },
                            }}
                            graphOptions={{
                              marginx: 15,
                              marginy: 15,
                              rankdir: 'TD',
                              ranksep: 55,
                              nodesep: 15,
                            }}
                          />
                        ) : null}
                      </svg>
                    </UncontrolledReactSVGPanZoom>
                  </div>
                </div>
              </Dropdown>
            </Card>
          </Col>
          <Col span={7} offset={1}>
            <Card bordered={false}>
              {/* TODO customize the config form */}
              <SchemaForm
                labelCol={8}
                wrapperCol={10}
                components={components}
                onSubmit={this.configSubmit}
                schema={this.state.configSchema}
                effects={({ setFieldState }) => {
                  merge(
                    onFieldValueChange$('dataset'),
                    onFieldInit$('dataset')
                  ).subscribe((fieldState) => {
                    setFieldState('upload', (state) => {
                      state.visible = fieldState.value === 'Customized';
                    });
                  });
                }}
              >
                <div
                  style={{
                    display:
                      this.state.isValidating || this.state.seconds === 0
                        ? ''
                        : 'none',
                  }}
                >
                  <Row>
                    <span>
                      Validation Accuracy :&nbsp;
                      {this.state.seconds === 0
                        ? `${0.0} %` // TODO replace with ajax data
                        : '  In Progress  '}
                    </span>
                  </Row>
                  <Row justify="center" style={{ width: '100%' }}>
                    <Progress
                      percent={100 - (this.state.seconds / 60) * 100}
                      format={() => `${this.state.seconds} s`}
                    />
                  </Row>
                </div>
                <FormButtonGroup offset={2}>
                  <Tooltip
                    placement="bottom"
                    title="Submit and run finetune Job"
                  >
                    <Submit
                      type="default"
                      size="large"
                      htmlType="submit"
                      style={{ width: 150 }}
                      disabled={this.state.isValidating}
                    >
                      Start Training
                    </Submit>
                  </Tooltip>
                  <Tooltip
                    placement="bottom"
                    title="Validate accuracy of modified model structure"
                  >
                    <Button
                      type="default"
                      size="large"
                      onClick={this.onClickValidate}
                      disabled={this.state.isValidating}
                      style={{ width: 150 }}
                    >
                      Fast Validate
                    </Button>
                  </Tooltip>
                </FormButtonGroup>
              </SchemaForm>
            </Card>
          </Col>
        </Row>
      </div>
    );
  }
}
