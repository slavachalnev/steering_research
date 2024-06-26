const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
	app.use(
		"/api",
		createProxyMiddleware({
			target: "https://api.replicate.com/v1/predictions",
			changeOrigin: true,
			pathRewrite: {
				"^/api": "",
			},
			onProxyReq: (proxyReq, req, res) => {
				console.log("hi");
				// Add headers here if needed
				proxyReq.setHeader(
					"Authorization",
					`Bearer ${process.env.REPLICATE_API_TOKEN}`
				);
			},
		})
	);
};
