# NourishWise API

NourishWise API is a backend service that utilises IBM's GRANITE_13B_CHAT_V2 to generate text-based responses based on user prompts. This service is designed for the NourishWise food recommendation application, which helps users find nearby restaurants with healthy food options tailored to their health conditions and dietary goals.

You don't have to install anything. The API is currently live. [Click to test api endpoint](https://nourishwise.onrender.com/generate)

Demo Video: [watch demo video](https://youtu.be/0l1rC3WXNs0)

Assistant Link: [Open Assistant Link](https://web-chat.global.assistant.watson.appdomain.cloud/preview.html?backgroundImageURL=https%3A%2F%2Fus-south.assistant.watson.cloud.ibm.com%2Fpublic%2Fimages%2Fupx-0cdfb006-d2f7-45c8-b1b7-b2b9b8e443bd%3A%3A00921169-c8fb-4a7b-90ba-b48cae577d0f&integrationID=f56651cf-5f9d-4db3-a679-42e72123db35&region=us-south&serviceInstanceID=0cdfb006-d2f7-45c8-b1b7-b2b9b8e443bd)

*** MAKE SURE YOU START THE ASSISTANT WITH THE "SURE" KEYWORD ***

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Licence](#licence)

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/nourishwise-api.git
   cd nourishwise-api
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your IBM Watson Machine Learning credentials:
   ```env
   APIKEY=your-ibm-watson-api-key
   PROJECT_ID=your-ibm-watson-project-id
   PORT=5000
   ```

## Usage

1. **Run the application:**
   ```sh
   python app.py
   ```

2. **Access the API:**
   The API will be available at `[link to api](https://nourishwise.onrender.com/generate)`.

## API Endpoints

### `POST /generate`

#### Description
Generate text based on the given prompt by sending it to the IBM Watson LLM.

#### Request
- **Headers:**
  - `Content-Type: application/json`
- **Body:**
  ```json
  {
    "prompt": "Your text prompt here"
  }
  ```

#### Responses

- **200 OK:**
  ```json
  {
    "response": "Generated text response"
  }
  ```
- **400 Bad Request:**
  ```json
  {
    "error": "Prompt is required"
  }
  ```

## Environment Variables

The application requires the following environment variables to be set:

- `APIKEY`: Your IBM Watson API key.
- `PROJECT_ID`: Your IBM Watson project ID.
- `PORT`: The port on which the Flask application will run (default: 5000).

## Licence

TODO
