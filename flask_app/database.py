import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# load_dotenv()
# URI_DB = os.getenv('URI_DB')
# db_engine = create_engine(URI_DB)

def query_db(db_name, var, db_engine):
    """Return a query for dep_variable from db_name."""
    if db_name == 'COVDB':
        query = text(
            f"""
                SELECT "covVar", "absTstat", "pValue", "significant"
                FROM "{db_name}"
                WHERE "depVar" = '{var}';
            """.format(db_name, var)
        )
        with db_engine.connect() as conn:
            results = (
                conn
                    .execute(query)
                    .fetchall()
            )
    elif db_name == 'GEODB':
        query1 = text(
            f"""
                SELECT "stateAbbr", "residChng", "depVarFirst", "firstYear", "depVarLast", "lastYear"
                FROM "{db_name}"
                WHERE "depVar" = '{var}'
                ORDER BY "depVarLast" DESC
                LIMIT 5;
            """.format(db_name, var)
        )
        query2 = text(
            f"""
                SELECT "stateAbbr", "residChng", "depVarFirst", "firstYear", "depVarLast", "lastYear"
                FROM "{db_name}"
                WHERE "depVar" = '{var}'
                ORDER BY "depVarLast" ASC
                LIMIT 5;
            """.format(db_name, var)
        )
        with db_engine.connect() as conn:
            results = (
                          conn
                              .execute(query1)
                              .fetchall()
                      ) + (
                          conn
                              .execute(query2)
                              .fetchall()
                      )
    elif db_name == 'CORRDB':
        query = text(
            f"""
                SELECT "indepVar", "beta", "pValue", "significant"
                FROM "{db_name}"
                WHERE "depVar" = '{var}'
                ORDER BY "pValue"
                LIMIT 10;
            """.format(db_name, var)
        )
        with db_engine.connect() as conn:
            results = (
                conn
                    .execute(query)
                    .fetchall()
            )
    elif db_name == 'TSDB':
        query = text(
            f"""
                SELECT "indep", "count", "state", "minPval"
                FROM "{db_name}"
                WHERE "depVar" = '{var}'
                LIMIT 10;
            """.format(db_name, var)
        )
        with db_engine.connect() as conn:
            results = (
                conn
                    .execute(query)
                    .fetchall()
            )

    return results
