<template>
  <div id="app">

    <input type="checkbox" id="predictorCheckbox" v-model="predictBoxTicked">
    <label for="predictorCheckbox">Predict with Saved Model</label>
    <Predictor v-if="predictBoxTicked" header="Ping Pong Bing Bong"/>
    <Training v-else header="Bing Bong Ping Pong"/>

  </div>
</template>

<script>
import Training from './components/Trainer.vue'
import Predictor from './components/Predictor.vue'

export default {
  name: 'App',
  data() {
    return {predictBoxTicked: false}
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
