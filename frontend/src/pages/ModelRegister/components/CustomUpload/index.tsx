import React from 'react';
import { Upload } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

const { Dragger } = Upload;

export default function CustomUpload(props) {
  return (
    <Dragger
    name={props.name}
    style={{display: 'block'}}
    onChange={
      (value)=>{
        let file = value.file;
        delete file.uid;
        props.onChange(file)
      }
    }
    beforeUpload={(file) => {
      return false;
    }
    }
    >
    <p className="ant-upload-drag-icon">
      <InboxOutlined />
    </p>
    <p className="ant-upload-text">Click or drag file to this area to upload</p>
    <p className="ant-upload-hint">
      Support for a single upload
    </p>
  </Dragger>
  );
}