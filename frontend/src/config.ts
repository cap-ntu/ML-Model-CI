// FIXME: unable to get environment variables from process.env
const host = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000'

export default {
  default: {
    modelURL: `${host}/api/v1/model`,
    visualizerURL: `${host}/api/v1/visualizer`,
    structureURL: `${host}/api/exp/structure`,
    structureRefractorURL: `${host}/api/exp/cv-tuner/finetune`, // temp url
    trainerURL: `${host}/api/exp/train`
  }
}
