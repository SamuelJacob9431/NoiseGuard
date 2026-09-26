NoiseGuard frontend

Deploy the contents of this folder to Netlify. index.html is the landing page; tool.html is the tool.

The tool calls the Kaggle Cloudflare Quick Tunnel URL in tool.js. If Kaggle restarts and the tunnel URL changes, update API_URL in tool.js and redeploy.

The Evaluation section now displays a downstream Stable Diffusion v1.5 Img2Img reconstruction returned by the backend, plus evaluation metrics. The layout is responsive for desktop, tablet, and mobile viewports.
