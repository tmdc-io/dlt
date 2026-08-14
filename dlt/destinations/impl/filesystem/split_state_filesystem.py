from typing import Iterable, Optional

from dlt.common.destination import DestinationCapabilitiesContext
from dlt.common.schema import Schema
from dlt.destinations import path_utils
from dlt.destinations.impl.filesystem.configuration import FilesystemDestinationClientConfiguration
from dlt.destinations.impl.filesystem.filesystem import FilesystemClient, INIT_FILE_NAME


class SplitStateFilesystemClient(FilesystemClient):
    """FilesystemClient that routes _dlt_* internal tables to a separate storage path.

    Enables destinations like Microsoft Fabric OneLake where user data must land in
    /Tables (Delta, SQL-queryable) while dlt internal state goes to /Files.
    """

    def __init__(
        self,
        schema: Schema,
        config: FilesystemDestinationClientConfiguration,
        capabilities: DestinationCapabilitiesContext,
    ) -> None:
        super().__init__(schema, config, capabilities)
        # _strip_protocol mirrors how bucket_path is derived via fsspec_from_config
        self._state_bucket_path: Optional[str] = (
            self.fs_client._strip_protocol(config.state_bucket_url)
            if config.state_bucket_url
            else None
        )

    @property
    def state_dataset_path(self) -> str:
        if not self._state_bucket_path:
            return self.dataset_path
        return self.pathlib.join(self._state_bucket_path, self.dataset_name, "")  # type: ignore[no-any-return]

    @property
    def init_file_path(self) -> str:
        return str(self.pathlib.join(self.state_dataset_path, INIT_FILE_NAME))

    def get_table_prefix(self, table_name: str) -> str:
        if self.is_dlt_table(table_name):
            table_prefix = self.pathlib.join(table_name, "")
            return self.pathlib.join(  # type: ignore[no-any-return]
                self.state_dataset_path,
                path_utils.normalize_path_sep(self.pathlib, table_prefix),
            )
        return super().get_table_prefix(table_name)

    def initialize_storage(self, truncate_tables: Iterable[str] = None) -> None:
        if self.state_dataset_path != self.dataset_path:
            self.fs_client.makedirs(self.state_dataset_path, exist_ok=True)
        super().initialize_storage(truncate_tables)

    def drop_storage(self) -> None:
        super().drop_storage()
        if self._state_bucket_path and self.state_dataset_path != self.dataset_path:
            if self.fs_client.exists(self.state_dataset_path):
                self.fs_client.rm(self.state_dataset_path, recursive=True)
