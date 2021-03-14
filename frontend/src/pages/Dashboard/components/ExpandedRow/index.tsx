import React from 'react';
import { Descriptions, Tag, Divider, Button} from 'antd'
export default class ExpandedRow extends React.Component{
  render(){
    return (
      <div style={{ backgroundColor: '#F5F5F5', padding: '10px' }}>
      <Descriptions
        style={{ width: '92%', margin: '0 auto' }}
        column={3}
        size="middle"
        title="Converted Model Info"
      >
        <Descriptions.Item
          label="Model Name"
        >
          <Tag color="volcano">{this.props.record.architecture}</Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="Converted Version"
        >
          <Tag color="blue">
            {this.props.record.engine === 'PYTORCH' ? '' : this.props.record.framework}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="Serving Engine"
        >
          <Tag color="pink">
            {(() => {
              switch (this.props.record.engine) {
                case 'TorchScript':
                  return 'PyTorch JIT + FastAPI';
                case 'ONNX':
                  return 'ONNX Runtime + FastAPI';
                case 'tensorrt':
                  return 'Triton Inference Server';
                case 'TFS':
                  return 'TensorFlow Serving';
                default:
                  return 'FastAPI';
              }
            })()}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          span={2}
          label="Conversion State"
        >
          <a
            style={{
              fontSize: 20,
              whiteSpace: 'nowrap',
              color: 'green',
            }}
          >
            {this.props.record.status}
          </a>
          <Divider type="vertical" />
          <Button type="primary" size="large">
            Deploy this Model
          </Button>
        </Descriptions.Item>
      </Descriptions>
      <Descriptions
        style={{ width: '92%', margin: '0 auto' }}
        column={3}
        size="small"
        title="Profiling Results"
      >
        <Descriptions.Item
          label="Profiling Device"
        >
          <Tag color="geekblue">
            {this.props.record.profile_result
              ? this.props.record.profile_result.dynamic_results[0].device_name
              : ''}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="Profiling Batch Size"
        >
          <Tag color="gold">
            {this.props.record.profile_result
              ? this.props.record.profile_result.dynamic_results[0].batch
              : ''}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="Device Memory Utilization"
        >
          <Tag color="blue">
            {this.props.record.profile_result
              ? (
                  this.props.record.profile_result.dynamic_results[0].memory
                    .utilization * 100
                ).toFixed(2)
              : ''}{' '}
            %
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="Preprocess Latency"
        >
          <Tag color="cyan">
            {this.props.record.profile_result
              ? (
                  this.props.record.profile_result.dynamic_results[0].latency
                    .preprocess_latency.avg * 1000
                ).toFixed(2)
              : ''}{' '}
            ms
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          span={2}
          label="All Batch Throughput"
        >
          <Tag color="geekblue">
            {this.props.record.profile_result
              ? this.props.record.profile_result.dynamic_results[0].throughput.inference_throughput.toFixed(
                  2
                )
              : ''}{' '}
            req/sec
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="P50 Latency"
        >
          <Tag color="gold">
            {this.props.record.profile_result
              ? (
                  this.props.record.profile_result.dynamic_results[0].latency
                    .inference_latency.p50 * 1000
                ).toFixed(2)
              : ''}{' '}
            ms
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="P95 Latency"
        >
          <Tag color="green">
            {this.props.record.profile_result
              ? (
                  this.props.record.profile_result.dynamic_results[0].latency
                    .inference_latency.p95 * 1000
                ).toFixed(2)
              : ''}{' '}
            ms
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item
          label="P99 Latency"
        >
          <Tag color="pink">
            {this.props.record.profile_result
              ? (
                  this.props.record.profile_result.dynamic_results[0].latency
                    .inference_latency.p99 * 1000
                ).toFixed(2)
              : ''}{' '}
            ms
          </Tag>
        </Descriptions.Item>
      </Descriptions>
    </div>
    )
  }
}
