'use strict';
const execBuffer = require('exec-buffer');
const isPng = require('is-png');
const zopfli = require('zopflipng-bin');

module.exports = options => buf => {
	options = {...options};

	if (!Buffer.isBuffer(buf)) {
		return Promise.reject(new TypeError('Expected a buffer'));
	}

	if (!isPng(buf)) {
		return Promise.resolve(buf);
	}

	const args = ['-y'];

	if (options['8bit']) {
		args.push('--lossy_8bit');
	}

	if (options.transparent) {
		args.push('--lossy_transparent');
	}

	if (options.iterations) {
		args.push(`--iterations=${options.iterations}`);
	}

	if (options.more) {
		args.push('-m');
	}

	args.push(execBuffer.input, execBuffer.output);

	return execBuffer({
		input: buf,
		bin: zopfli,
		args
	}).catch(error => {
		error.message = error.stderr || error.message;
		throw error;
	});
};
