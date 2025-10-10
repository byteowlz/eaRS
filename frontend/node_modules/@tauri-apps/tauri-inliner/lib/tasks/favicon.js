module.exports = resolve
const debug = require('debug')('inliner')
const basename = require('path').basename

function resolve (inliner, todo, $) {
  debug('start %s favicon', todo.length)
  return todo.map(function links (link) {
    let url = $(link).attr('href')
    url = inliner.resolve(inliner.url, url)
    inliner.emit('progress', 'processing favicon ' + basename(url))
    return inliner.image(url).then(function then (dataURL) {
      $(link).attr('href', dataURL)
    }).then(inliner.jobs.done.favicon)
  })
}
