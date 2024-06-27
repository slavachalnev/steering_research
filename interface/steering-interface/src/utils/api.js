export const steeringData = {
	10138: "London",
	2378: "Wedding",
	11067: "Poetry",
};

export const predictData = async (data) => {
	const url = "http://192.168.0.202:5000/predict";

	try {
		// Sending the initial request to the /predict endpoint
		let response = await fetch(url, {
			method: "POST",
			headers: {
				// Authorization: `Bearer ${api_key}`,
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});

		if (!response.ok) {
			throw new Error(`Initial request failed: ${response.statusText}`);
		}

		let responseData = await response.json();

		// Extracting the prediction ID for polling
		let predictionId = responseData.id;
		let status = responseData.status;

		// Polling the API until the prediction is complete
		while (status !== "succeeded" && status !== "failed") {
			await new Promise((resolve) => setTimeout(resolve, 5000)); // Wait for 5 seconds before polling again

			let pollUrl = `${url}/${predictionId}`;
			response = await fetch(pollUrl, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
			});

			if (!response.ok) {
				throw new Error(`Polling request failed: ${response.statusText}`);
			}

			responseData = await response.json();
			status = responseData.status;
			console.log(`Prediction status: ${status}`);
		}

		// Return the final response
		if (status === "succeeded") {
			return {
				status: "succeeded",
				output: responseData.output,
			};
		} else {
			throw new Error(`Prediction failed: ${responseData.error}`);
		}
	} catch (error) {
		console.error("Error:", error);
		return {
			status: "failed",
			error: error.message,
		};
	}
};

export const getNeuronpedia = async (feature) => {
	fetch(
		`https://www.neuronpedia.org/api/feature/gemma-2b/6-res-jb/${feature}`,
		{
			method: "GET",
			headers: {
				// "Content-Type": "application/json",
				Accept: "*/*",
			},
		}
	)
		.then((res) => res.json())
		.then((data) => {
			console.log(data);
			return data;
		})
		.catch((error) => {
			console.log(error);
			return error;
		});
};
