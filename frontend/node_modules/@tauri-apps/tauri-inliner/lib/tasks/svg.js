module.exports = resolve

const Promise = require('es6-promise').Promise // jshint ignore:line
const debug = require('debug')('inliner')
const SVGO = require('svgo')
const svgo = new SVGO({
  plugins: [{
    cleanupIDs: false
  }]
})

function resolve (inliner, todo, $) {
  debug('start %s svg', todo.length, !!$)
  return todo.map(function (svg) {
    if (inliner.options.nosvg) {
      return Promise.resolve()
    }

    const $svg = $(svg)
    let source = $svg.html()

    debug('optimising svg')

    // reconstruct the SVG element outer tag
    // console.log(Object.keys(svg.attribs).map(_ => `${_}="${svg.attribs[_]}"`));
    const top = '<svg ' + Object.keys(svg.attribs).map(function (key) {
      // eslint-disable-next-line
      return key + '="' + svg.attribs[key] + '"'
    }).join(' ') + '>'

    source = top + source + '</svg>'

    return svgo.optimize(source).then(function (result) {
      if (result.error) {
        debug('svg failed', result.error)
        return
      }
      debug('optimisation complete')

      $svg.replaceWith(result.data)
    }).then(inliner.jobs.done.svg)
  })
}
