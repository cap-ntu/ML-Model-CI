const host = 'http://155.69.146.35:8000'
export default {
  default: {
    modelURL: `${host}/api/v1/model`,
    visualizerURL: `${host}/api/v1/visualizer`,
    structureURL: `${host}/api/exp/structure`,
    structureRefractorURL: `${host}/api/exp/cv-tuner/finetune`, // temp url
    trainerURL: `${host}/api/exp/train`
  }
}
