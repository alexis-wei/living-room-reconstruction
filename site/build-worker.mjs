import {mkdir,copyFile} from 'node:fs/promises';
await mkdir('dist/server',{recursive:true});
await mkdir('dist/.openai',{recursive:true});
await copyFile('worker.js','dist/server/index.js');
await copyFile('.openai/hosting.json','dist/.openai/hosting.json');
