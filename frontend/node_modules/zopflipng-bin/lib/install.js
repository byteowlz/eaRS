'use strict';
const path = require('path');
const binBuild = require('bin-build');
const log = require('logalot');
const bin = require('.');

bin.run(['--help']).then(() => {
	log.success('zopflipng pre-build test passed successfully');
}).catch(async error => {
	log.warn(error.message);
	log.warn('zopflipng pre-build test failed');
	log.info('compiling from source');

	let makeBin = 'make';
	let makeArgs = '';

	if (process.platform === 'freebsd') {
		makeBin = 'gmake';
		makeArgs = 'CC=cc CXX=c++';
	}

	try {
		await binBuild.file(path.resolve(__dirname, '../vendor/source/zopfli-1.0.3.tar.gz'), [
			`mkdir -p ${bin.dest()}`,
			`${makeBin} zopflipng ${makeArgs} && mv ./zopflipng ${bin.path()}`
		]);

		log.success('zopflipng built successfully');
	} catch (error) {
		log.error(error.stack);
	}
});
