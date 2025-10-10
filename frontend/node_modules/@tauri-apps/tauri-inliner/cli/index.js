#!/usr/bin/env node

const readFileSync = require('fs').readFileSync
const Promise = require('es6-promise').Promise // jshint ignore:line
const ansi = require('ansi-escapes')
const path = require('path')

main()

function main () {
  const argv = require('./options')(process.argv.slice(2))

  if (argv.debug) {
    require('debug').enable('inliner')
  }

  const url = argv._.shift()

  if (argv.version) {
    console.log(require('../package.json').version || 'development')
    process.exit(0)
  }

  if ((!url && !argv.useStdin) || argv.help) {
    const usage = readFileSync(
      path.join(__dirname, '../docs/usage.txt'), 'utf8'
    )
    console.log(usage)
    process.exit(0)
  }

  const Inliner = require('../')

  let p = Promise.resolve(url)

  if (argv.useStdin) {
    p = new Promise(function (resolve) {
      process.stdin.resume()
      process.stdin.setEncoding('utf8')

      let data = ''

      process.stdin.on('data', function (chunk) {
        data += chunk
      })

      process.stdin.on('end', function () {
        resolve(data)
      })
    })
  }

  const time = process.hrtime()
  p.then(function (source) {
    const inliner = new Inliner(source, argv, function result (error, html) {
      if (error) {
        const message = Inliner.errors[error.code] || error.message
        console.error(message)

        if (argv.debug) {
          console.error(error.stack)
        }
        process.exit(1)
      }

      console.log(html)
    })

    return inliner
  }).then(function (inliner) {
    // checks for available update and returns an instance
    // note: we're doing this after we kick off inliner, since there's a
    // noticeable lag in boot because of it
    const defaults = require('lodash').defaults
    const pkg = JSON.parse(readFileSync(path.join(__dirname, '../package.json')))

    require('update-notifier')({
      pkg: defaults(pkg, { version: '0.0.0' })
    }).notify()

    if (argv.verbose) {
      let jobs = {}
      const update = require('./progress')
      let progress = ''

      inliner.on('progress', function progressEvent (event) {
        progress = event
        // console.log(JSON.stringify({ type: 'progress', progress, jobs }));
        update(event, jobs, argv.debug)
      })

      inliner.on('jobs', function jobsEvent (event) {
        jobs = event
        // console.log(JSON.stringify({ type: 'jobs', progress, jobs }));
        update(progress, jobs, argv.debug)
      })

      inliner.on('warning', function warningEvent (event) {
        progress = event
        // console.log(JSON.stringify({ type: 'warning', progress, jobs }));
        update(event, jobs, true)
      })

      inliner.on('end', function () {
        update.end(time)
      })

      'exit SIGINT SIGTERM'.split(' ').map(function (event) {
        process.once(event, function () {
          process.stderr.write(ansi.cursorShow) // put the cursor back
          try { process.kill(process.pid, event) } catch (e) {}
        })
      })
    } else {
      inliner.on('warning', function progress (event) {
        console.warn('warning: ' + event)
      })
    }
  }).catch(function (error) {
    console.error(error.stack)
  })
}
