<template>
  <div id="app">

    <input type="checkbox" id="predictorCheckbox" v-model="predictBoxTicked">
    <label for="predictorCheckbox">Predict with Saved Model</label>
    <Predictor 
      v-if="predictBoxTicked" 
      header="Model Inference" 
      v-bind:w="inputSize.w" 
      v-bind:h="inputSize.h"
    />
    <Training 
      v-else
      header="Model Training" 
      v-bind:w="inputSize.w" 
      v-bind:h="inputSize.h"
    />

  </div>
</template>

<script>
import Training from './components/Trainer.vue'
import Predictor from './components/Predictor.vue'

export default {
  name: 'App',
  data() {
    return {
      inputSize: {w:224, h:224}, // previously {w:512, h:512}
      // flags vvv
      predictBoxTicked: false
    }
  },
  components: {
    Training,
    Predictor
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
