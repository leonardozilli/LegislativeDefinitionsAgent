# LegalDefAgent

## Installation

1. Clone the repository:

   ```sh
    git clone https://github.com/leonardozilli/LegalDefAgent.git

    cd LegalDefAgent
   ```

2. Create a virtual environment
    ```sh
    python -m venv .venv

    source .venv/bin/activate
    ```

3. Install the dependencies and module:

   ```sh
    pip install -r requirements.txt

    pip install .
    ```

## Usage

1. Rename the `.env-example` to `.env` and populate the file with the required credentials


2. Run the FastAPI server:

   ```sh
   python scripts/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run scripts/streamlit_app.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.