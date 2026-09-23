import glob

from absl import logging
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from aequilibrae.matrix import AequilibraeMatrix
from shapely.geometry import Point
from sqlalchemy import create_engine


def read_data(dataset_name: str, cwd: str=None):
    if cwd:
        data_root_path = os.path.join(cwd, "data", "raw")
    else:
        data_root_path = os.path.join(os.getcwd(), "data", "raw")
    network_file = f"{data_root_path}/{dataset_name}/transportation_resilience_network.csv"
    demand_file = f"{data_root_path}/{dataset_name}/transportation_resilience_od.csv"
    aemfile = f"{data_root_path}/{dataset_name}/{dataset_name}.aem"

    dem = pd.read_csv(demand_file)
    zones = int(max(dem.O.max(), dem.D.max()))
    index = np.arange(zones) + 1
    mtx = np.zeros(shape=(zones, zones))
    for element in dem.to_records(index=False):
        mtx[element[0] - 1][element[1] - 1] = element[2]
    aem = AequilibraeMatrix()
    kwargs = {"file_name": aemfile,
              "zones": zones,
              "matrix_names": ["matrix"]}
    aem.create_empty(**kwargs)
    aem.matrix["matrix"][:, :] = mtx[:, :]
    aem.index[:] = index[:]
    network = pd.read_csv(network_file, index_col=0)

    if (network["capacity"] <= 0).any():
        logging.debug("Sanity test on capacity failed: at least one link with zero or negative capacity.")
        logging.debug("Set capacity as 0.01 for failed links.")
        network["capacity"] = network["capacity"].clip(lower=0.01)
    if (network["power"] < 1).any():
        logging.debug("Sanity test on power failed: at least one link with power less than 1.")
        logging.debug("Set power as 4.0 for failed links.")
        network["power"] = network["power"].apply(lambda x: 4.0 if x<1 else x)
    if (network["free_flow_time"] <= 0).any():
        logging.debug("Sanity test on free_flow_time failed: at least one link with zero or negative free_flow_time.")
        logging.debug("Set free_flow_time as 0.01 for failed links.")
        network["free_flow_time"] = network["free_flow_time"].clip(lower=0.01)
    if (network["free_flow_time"].isna()).any():
        logging.debug("Sanity test on free_flow_time failed: at least one link with missing free_flow_time.")
        logging.debug("Set free_flow_time as 0.01 for failed links.")
        network["free_flow_time"] = network["free_flow_time"].fillna(0.01)
    if dataset_name == 'Munich':
        invalid = (network["free_flow_time"] == np.inf) & (network['length'] == 0.0)
        network['free_flow_time'] = network['free_flow_time'].mask(invalid, 0.01).clip(upper=10000)
    if dataset_name == "Winnipeg-Asymmetric":
        # 这个数据集中只有 network.a_node.min() == len(index) + 1
        # 说明只有从 non-zone 到 zone 的 link，没有从 zone 到 non-zone 的 link
        # 所以需要添加一些反向的 link
        add = network[network.b_node.isin(index)].copy()
        add.a_node, add.b_node = add.b_node, add.a_node
        network = pd.concat([network, add], ignore_index=True)
        network.link_id = network.index + 1

    return aem, network, index


def save_result_to_db(df: pd.DataFrame, table_name: str):
    save_result_to_sqlite(df, table_name)
    save_result_to_postgres(df, table_name)


def save_result_to_sqlite(df: pd.DataFrame, table_name: str):
    try:
        logging.info(f"Saving {len(df)} records to SQLite...")
        db_path = os.path.join(os.getcwd(), "db", "results.db")
        engine = create_engine(f"sqlite:///{db_path}")
        df.to_sql(table_name, engine, if_exists="append", index=False)
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving to SQLite: {e}")


def save_result_to_postgres(df: pd.DataFrame, table_name: str):
    pgsecret = os.getenv("PGSECRET")
    if pgsecret is None:
        raise ValueError("Environment variable 'PGSECRET' not found!")
    try:
        logging.info(f"Saving {len(df)} records to PostgreSQL...")
        engine = create_engine(f"postgresql://yu_zheng:{pgsecret}@pg-azure.postgres.database.azure.com/postgres")
        df.to_sql(table_name, engine, if_exists="append", index=False)
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving to PostgreSQL: {e}")


def test_postgres_connection(return_engine=False):
    pgsecret = os.getenv("PGSECRET")
    if pgsecret is None:
        raise ValueError("Environment variable 'PGSECRET' not found!")
    try:
        logging.info("Testing PostgreSQL connection...")
        engine = create_engine(f"postgresql://yu_zheng:{pgsecret}@pg-azure.postgres.database.azure.com/postgres")
        conn = engine.connect()
        logging.info("Connection successful!")
        conn.close()
        del conn
    except Exception as e:
        raise RuntimeError(f"An error occurred while connecting to PostgreSQL: {e}")
    if return_engine:
        return engine
    else:
        del engine


def read_nodes(dataset_name, data_root_path):
    if dataset_name in ["Birmingham-England"]:
        file_path = f"{data_root_path}/Birmingham-England/Birmingham_Nodes.tntp"
        df = pd.read_csv(file_path, skiprows=1, sep=r'\s+', names=["node_id", "x", "y"])
    elif dataset_name in ["Berlin-Prenzlauerberg-Center", "Berlin-Friedrichshain"]:
        dir_path = f"{data_root_path}/{dataset_name}/"
        file_path = glob.glob(dir_path + "*node.tntp")
        if len(file_path) != 1:
            raise RuntimeError(f"Found more or less than one geometry file: {file_path}")
        file_path = file_path[0]
        df = pd.read_csv(file_path, skiprows=1, sep=r'\s+', names=["node_id", "x", "y", ";"])
        df = df[["node_id", "x", "y"]]
    elif dataset_name in ["Valledupar", "SantaMarta", "Cartagena"]:
        file_path = f"{data_root_path}/{dataset_name}/transportation_resilience_node.csv"
        df = pd.read_csv(file_path)
        df = df[["remap_node_id", "x_coord", "y_coord"]].rename(
            columns={
                "remap_node_id": "node_id",
                "x_coord": "x",
                "y_coord": "y"})
    else:
        raise ValueError(f"Dataset {dataset_name} not supported!")
    df['geometry'] = df.apply(lambda row: Point(row["x"], row["y"]), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    return gdf

