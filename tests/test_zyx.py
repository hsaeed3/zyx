# full library test runnable
# if you dont want to use pytest, you can run this file directly

from tests import (
    test_logging,
    test_api_resource
)


# ========================================================================
# [Logging]
# ========================================================================

test_logging.test_logging()


# ========================================================================
# [API Resource]
# ========================================================================

test_api_resource.test_api_resource_litellm_resource()
test_api_resource.test_api_resource_init()
test_api_resource.test_api_resource_instructor_patch()
