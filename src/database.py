from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

def get_connection():
    hostname = 'localhost'
    database = 'telecom_db'
    username = 'postgres'
    pwd = '2525'
    port_id = '5432'

    # Construct the database URL required by SQLAlchemy
    db_url = f'postgresql+psycopg2://{username}:{pwd}@{hostname}:{port_id}/{database}'

    try:
        # Create an SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Test the connection by establishing it temporarily
        with engine.connect() as connection:
            print("Connection successful")
        
        # Return the engine to be used with pandas
        return engine
    except OperationalError as e:
        print("OperationalError:", e)
    except Exception as e:
        print("Error connecting to the database:", e)
    return None

